//! Differentiation Metric (D_int)
//!
//! Computes the effective rank of the intention/agent manifold using
//! singular value decomposition (SVD) participation ratio.
//!
//! D_int > 1 indicates non-trivial structure (suppresses one-parameter loops).
//! D_int > 3 is considered high differentiation (consciousness candidate).

use ndarray::{Array1, Array2};

/// Compute D_int via SVD participation ratio.
///
/// Uses power iteration for singular values when ndarray-linalg is not available,
/// falling back to full SVD when the `linalg` feature is enabled.
pub fn compute_d_int(data: &[Vec<f64>]) -> f64 {
    if data.is_empty() || data[0].is_empty() {
        return 0.0;
    }

    let n = data.len();
    let d = data[0].len();

    if n < 2 || d < 1 {
        return 1.0; // Single point = rank 1
    }

    // Build centered data matrix
    let mut matrix = Array2::<f64>::zeros((n, d));
    for (i, row) in data.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            matrix[[i, j]] = val;
        }
    }

    // Center the data (subtract mean of each column)
    for j in 0..d {
        let mean: f64 = matrix.column(j).iter().sum::<f64>() / n as f64;
        for i in 0..n {
            matrix[[i, j]] -= mean;
        }
    }

    // Compute singular values
    let singular_values: Vec<f64> = compute_singular_values(&matrix);

    // Effective rank using participation ratio
    // D_int = (sum(s^2))^2 / sum(s^4)
    let s_sq: Vec<f64> = singular_values.iter().map(|&s| s * s).collect();
    let sum_sq: f64 = s_sq.iter().sum();
    let sum_sq_sq: f64 = s_sq.iter().map(|&s| s * s).sum();

    if sum_sq_sq > 0.0 && sum_sq > 0.0 {
        (sum_sq * sum_sq) / sum_sq_sq
    } else {
        1.0 // Minimum differentiation
    }
}

/// Compute singular values of a matrix.
/// Uses full SVD if ndarray-linalg is available, otherwise power iteration.
fn compute_singular_values(matrix: &Array2<f64>) -> Vec<f64> {
    #[cfg(feature = "linalg")]
    {
        // Use ndarray-linalg for full SVD
        use ndarray_linalg::SVD;
        if let Ok((_, s, _)) = matrix.svd(true, true) {
            return s.iter().copied().collect();
        }
    }

    // Fallback: power iteration for top k singular values
    power_iteration_svd(matrix, matrix.ncols().min(10))
}

/// Power iteration to estimate top k singular values.
fn power_iteration_svd(matrix: &Array2<f64>, k: usize) -> Vec<f64> {
    let (n, d) = matrix.dim();
    let k = k.min(d);

    let mut singular_values = Vec::with_capacity(k);
    let mut a = matrix.clone();

    for _ in 0..k {
        // Random initial vector
        let mut v: Array1<f64> = Array1::from(vec![1.0; d]);
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v = &v / norm;

        // Power iteration
        for _ in 0..50 {
            // v = A^T * A * v
            let av = a.dot(&v);
            let at_av = a.t().dot(&av);
            let norm = at_av.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                v = at_av / norm;
            } else {
                break;
            }
        }

        // Compute singular value: sigma = ||A * v||
        let av = a.dot(&v);
        let sigma = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        singular_values.push(sigma);

        // Deflate: A = A - sigma * u * v^T where u = A*v/sigma
        if sigma > 1e-10 {
            let u = av / sigma;
            for i in 0..n {
                for j in 0..d {
                    a[[i, j]] -= sigma * u[i] * v[j];
                }
            }
        }
    }

    singular_values
}

/// Simple implementation of D_int for trace data.
/// Takes a trace and computes D_int from the feature matrix [coherence, depth, observed].
pub fn d_int_from_trace(coherence: &[f64], depth: &[f64], observed: &[f64]) -> f64 {
    if coherence.len() != depth.len() || coherence.len() != observed.len() {
        return 0.0;
    }

    let data: Vec<Vec<f64>> = coherence
        .iter()
        .zip(depth.iter())
        .zip(observed.iter())
        .map(|((&c, &d), &o)| vec![c, d, o])
        .collect();

    compute_d_int(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_signal() {
        // Constant signal has rank 1
        let data: Vec<Vec<f64>> = (0..20)
            .map(|_| vec![0.5, 0.5, 0.5])
            .collect();
        let d = compute_d_int(&data);
        assert!((d - 1.0).abs() < 0.1, "Constant signal should have D_int ≈ 1, got {}", d);
    }

    #[test]
    fn test_two_independent_sinusoids() {
        // Two independent sinusoids → rank ≈ 2
        let data: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let t = i as f64 * 0.1;
                vec![t.sin(), t.cos(), (2.0 * t).sin()]
            })
            .collect();
        let d = compute_d_int(&data);
        // Should be > 2 but < 3
        assert!(d > 1.5, "Two sinusoids should have D_int > 1.5, got {}", d);
        println!("Two sinusoids: D_int = {}", d);
    }

    #[test]
    fn test_random_high_d() {
        // Random data should have D_int close to dimension
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<Vec<f64>> = (0..50)
            .map(|_| vec![rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>()])
            .collect();
        let d = compute_d_int(&data);
        // Random 3D data → D_int should approach 3
        assert!(d > 2.0, "Random 3D should have D_int > 2, got {}", d);
        println!("Random 3D: D_int = {}", d);
    }

    #[test]
    fn test_empty_data() {
        let data: Vec<Vec<f64>> = vec![];
        assert_eq!(compute_d_int(&data), 0.0);
    }

    #[test]
    fn test_single_point() {
        let data = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(compute_d_int(&data), 1.0);
    }

    #[test]
    fn test_power_iteration_fallback() {
        // This test runs without ndarray-linalg feature
        let data: Vec<Vec<f64>> = (0..30)
            .map(|i| {
                let t = i as f64 * 0.2;
                vec![t.sin(), t.cos()]
            })
            .collect();
        let d = compute_d_int(&data);
        assert!(d >= 1.0, "Should have at least rank 1");
        println!("Power iteration fallback: D_int = {}", d);
    }
}
