//! Streaming reader for row-major `f32` `.npy` matrices.
//!
//! [`read_npy_f32_as`] converts the matrix to a dense dataset of value type `V` chunk by chunk,
//! so its peak memory is the converted dataset plus one chunk. Reading the whole `f32` matrix
//! first and converting it afterwards (as `vectorium::readers::read_npy_f32` followed by a
//! conversion does) needs both copies at once, which does not fit in memory for very large
//! collections (e.g., 138M x 1024 `f32` is 567 GB, its `f16` copy 283 GB).
//!
//! On Linux, each chunk is dropped from the page cache once read (`posix_fadvise` with
//! `POSIX_FADV_DONTNEED`): otherwise the file's cached pages fill the NUMA nodes the dataset
//! should be placed on, and the kernel places it on other nodes instead of reclaiming them.

use std::fs::File;
use std::io::{self, BufReader, Read};

use rayon::prelude::*;
use vectorium::encoders::dense_scalar::{PlainDenseQuantizer, ScalarDenseSupportedDistance};
use vectorium::{DenseDataset, Float, FromF32, ValueType};

/// Rows converted per chunk.
const CHUNK_ROWS: usize = 1 << 16;

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Reads the header of a 2-D little-endian `f32` C-order `.npy` file. Returns `(rows, cols)`.
fn read_header(r: &mut impl Read, path: &str) -> io::Result<(usize, usize)> {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err(invalid(format!("{path}: not a .npy file")));
    }
    let header_len = match magic[6] {
        1 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            u16::from_le_bytes(b) as usize
        }
        2 | 3 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)?;
            u32::from_le_bytes(b) as usize
        }
        v => return Err(invalid(format!("{path}: unsupported .npy version {v}"))),
    };
    let mut header = vec![0u8; header_len];
    r.read_exact(&mut header)?;
    let header = String::from_utf8_lossy(&header);
    if !header.contains("'descr': '<f4'") {
        return Err(invalid(format!("{path}: expected dtype '<f4', header {header}")));
    }
    if !header.contains("'fortran_order': False") {
        return Err(invalid(format!("{path}: expected C order, header {header}")));
    }
    let shape = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| invalid(format!("{path}: no shape in header {header}")))?;
    let dims: Vec<usize> = shape
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|_| invalid(format!("{path}: bad shape {shape}"))))
        .collect::<io::Result<_>>()?;
    match dims[..] {
        [rows, cols] => Ok((rows, cols)),
        _ => Err(invalid(format!("{path}: expected a 2-D matrix, shape ({shape})"))),
    }
}

/// Reads a row-major `f32` `.npy` matrix into a dense dataset of value type `V` (e.g., `f16`),
/// converting with [`FromF32::from_f32_saturating`] as `hnsw_build --value-type` does.
pub fn read_npy_f32_as<V, D>(path: &str) -> io::Result<DenseDataset<PlainDenseQuantizer<V, D>>>
where
    V: ValueType + Float + FromF32 + Send,
    D: ScalarDenseSupportedDistance,
{
    let file = File::open(path)?;
    #[cfg(target_os = "linux")]
    let fd = std::os::unix::io::AsRawFd::as_raw_fd(&file);
    let mut r = BufReader::with_capacity(1 << 24, file);
    let (rows, cols) = read_header(&mut r, path)?;
    let mut data: Vec<V> = Vec::with_capacity(rows * cols);
    let mut bytes = vec![0u8; CHUNK_ROWS * cols * 4];
    let mut done = 0;
    while done < rows {
        let n = CHUNK_ROWS.min(rows - done);
        let buf = &mut bytes[..n * cols * 4];
        r.read_exact(buf)?;
        let start = data.len();
        data.resize(start + n * cols, V::from_f32_saturating(0.0));
        data[start..]
            .par_iter_mut()
            .zip(buf.par_chunks_exact(4))
            .for_each(|(v, b)| *v = V::from_f32_saturating(f32::from_le_bytes(b.try_into().unwrap())));
        done += n;
        #[cfg(target_os = "linux")]
        // SAFETY: plain syscall on a descriptor owned by `r`; it only drops clean cached pages.
        unsafe {
            libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
    let mut rest = [0u8; 1];
    if r.read(&mut rest)? != 0 {
        return Err(invalid(format!("{path}: trailing bytes after {rows} x {cols} values")));
    }
    Ok(DenseDataset::from_raw(data.into_boxed_slice(), rows, PlainDenseQuantizer::new(cols)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use std::io::Write;
    use vectorium::Dataset;
    use vectorium::distances::DotProduct;

    fn write_npy(path: &std::path::Path, rows: usize, cols: usize, vals: &[f32]) {
        let mut h = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(' ');
        }
        h.push('\n');
        let mut f = File::create(path).unwrap();
        f.write_all(b"\x93NUMPY\x01\x00").unwrap();
        f.write_all(&(h.len() as u16).to_le_bytes()).unwrap();
        f.write_all(h.as_bytes()).unwrap();
        for v in vals {
            f.write_all(&v.to_le_bytes()).unwrap();
        }
    }

    #[test]
    fn reads_and_converts_across_chunks() {
        let (rows, cols) = (CHUNK_ROWS + 3, 2);
        let vals: Vec<f32> = (0..rows * cols).map(|i| (i % 1000) as f32 * 0.25 - 7.0).collect();
        let path = std::env::temp_dir().join(format!("kannolo_npy_test_{}.npy", std::process::id()));
        write_npy(&path, rows, cols, &vals);
        let ds = read_npy_f32_as::<f16, DotProduct>(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(ds.len(), rows);
        let expect: Vec<f16> = vals.iter().map(|&x| f16::from_f32_saturating(x)).collect();
        assert_eq!(ds.values(), &expect[..]);
    }
}
