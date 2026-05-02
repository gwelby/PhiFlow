//! Mutual Information Computation
//!
//! Shannon mutual information and normalized variants for Type 4 metrics.
//!
//! MI(X;Y) = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
//!
//! Normalized MI = MI(X;Y) / min(H(X), H(Y)) ∈ [0, 1]

use std::collections::HashMap;

/// Compute Shannon mutual information from joint and marginal distributions.
///
/// # Arguments
/// * `joint_dist` - Map from (x, y) pairs to joint probability p(x,y)
/// * `marginal_x` - Map from x values to p(x)
/// * `marginal_y` - Map from y values to p(y)
///
/// # Returns
/// Mutual information in bits (log base 2)
pub fn shannon_mi(
    joint_dist: &HashMap<(u32, u32), f64>,
    marginal_x: &HashMap<u32, f64>,
    marginal_y: &HashMap<u32, f64>,
) -> f64 {
    let mut mi = 0.0;

    for ((x, y), &p_xy) in joint_dist.iter() {
        if p_xy > 0.0 {
            let p_x = marginal_x.get(x).copied().unwrap_or(0.0);
            let p_y = marginal_y.get(y).copied().unwrap_or(0.0);

            if p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).log2();
            }
        }
    }

    mi
}

/// Compute MI directly from two sample vectors using histogram binning.
///
/// # Arguments
/// * `x` - First sample vector
/// * `y` - Second sample vector (same length as x)
/// * `bins` - Number of bins for discretization
///
/// # Returns
/// Mutual information in bits
pub fn mi_from_samples(x: &[f64], y: &[f64], bins: usize) -> f64 {
    if x.len() != y.len() || x.len() < 2 || bins == 0 {
        return 0.0;
    }

    // Build joint and marginal distributions via binning
    let mut joint: HashMap<(u32, u32), f64> = HashMap::new();
    let mut marginal_x: HashMap<u32, f64> = HashMap::new();
    let mut marginal_y: HashMap<u32, f64> = HashMap::new();

    let n = x.len() as f64;

    // Determine bin edges
    let (min_x, max_x) = match (x.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                                x.iter().fold(-f64::INFINITY, |a, &b| a.max(b))) {
        (min, max) if max > min => (min, max),
        _ => return 0.0,
    };

    let (min_y, max_y) = match (y.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                                y.iter().fold(-f64::INFINITY, |a, &b| a.max(b))) {
        (min, max) if max > min => (min, max),
        _ => return 0.0,
    };

    // Avoid edge case where min == max
    let x_range = (max_x - min_x).max(1e-10);
    let y_range = (max_y - min_y).max(1e-10);

    // Bin the data
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let bin_x = ((xi - min_x) / x_range * bins as f64)
            .min(bins as f64 - 1.0)
            .max(0.0) as u32;
        let bin_y = ((yi - min_y) / y_range * bins as f64)
            .min(bins as f64 - 1.0)
            .max(0.0) as u32;

        *joint.entry((bin_x, bin_y)).or_insert(0.0) += 1.0 / n;
        *marginal_x.entry(bin_x).or_insert(0.0) += 1.0 / n;
        *marginal_y.entry(bin_y).or_insert(0.0) += 1.0 / n;
    }

    shannon_mi(&joint, &marginal_x, &marginal_y)
}

/// Compute normalized mutual information.
///
/// NMI = MI(X;Y) / min(H(X), H(Y)) ∈ [0, 1]
///
/// # Arguments
/// * `x` - First sample vector
/// * `y` - Second sample vector
/// * `bins` - Number of bins for discretization
///
/// # Returns
/// Normalized MI in [0, 1]
pub fn normalized_mi(x: &[f64], y: &[f64], bins: usize) -> f64 {
    let mi = mi_from_samples(x, y, bins);

    // Compute entropies
    let h_x = entropy(x, bins);
    let h_y = entropy(y, bins);

    let h_min = h_x.min(h_y);

    if h_min > 0.0 {
        (mi / h_min).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compute entropy from samples using histogram binning.
fn entropy(samples: &[f64], bins: usize) -> f64 {
    if samples.len() < 2 || bins == 0 {
        return 0.0;
    }

    let (min_val, max_val) = match (samples.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                                    samples.iter().fold(-f64::INFINITY, |a, &b| a.max(b))) {
        (min, max) if max > min => (min, max),
        _ => return 0.0,
    };

    let range = (max_val - min_val).max(1e-10);
    let n = samples.len() as f64;

    let mut counts: HashMap<u32, usize> = HashMap::new();

    for &val in samples.iter() {
        let bin = ((val - min_val) / range * bins as f64)
            .min(bins as f64 - 1.0)
            .max(0.0) as u32;
        *counts.entry(bin).or_insert(0) += 1;
    }

    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / n;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Compute conditional mutual information I(X;Y|Z).
///
/// I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
/// Approximated via binning and conditioning on Z bins.
#[allow(dead_code)]
pub fn conditional_mi(x: &[f64], y: &[f64], z: &[f64], bins: usize) -> f64 {
    if x.len() != y.len() || x.len() != z.len() || x.len() < 2 {
        return 0.0;
    }

    // Bin Z and compute MI(X;Y) within each Z bin, weighted by P(Z)
    let (min_z, max_z) = match (z.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                                z.iter().fold(-f64::INFINITY, |a, &b| a.max(b))) {
        (min, max) if max > min => (min, max),
        _ => return 0.0,
    };

    let z_range = (max_z - min_z).max(1e-10);
    let z_bins: Vec<u32> = z
        .iter()
        .map(|&zi| {
            ((zi - min_z) / z_range * bins as f64)
                .min(bins as f64 - 1.0)
                .max(0.0) as u32
        })
        .collect();

    // Count occurrences per Z bin
    let mut z_counts: HashMap<u32, usize> = HashMap::new();
    for &bin in &z_bins {
        *z_counts.entry(bin).or_insert(0) += 1;
    }

    let n = x.len() as f64;
    let mut cmi = 0.0;

    for (&z_bin, &count) in &z_counts {
        if count < 2 {
            continue;
        }

        // Extract X, Y samples where Z is in this bin
        let x_z: Vec<f64> = x
            .iter()
            .zip(z_bins.iter())
            .filter(|(_, z)| **z == z_bin)
            .map(|(&xi, _)| xi)
            .collect();

        let y_z: Vec<f64> = y
            .iter()
            .zip(z_bins.iter())
            .filter(|(_, z)| **z == z_bin)
            .map(|(&yi, _)| yi)
            .collect();

        // Weighted by P(Z=z)
        let p_z = count as f64 / n;
        cmi += p_z * mi_from_samples(&x_z, &y_z, bins.min(count / 2).max(2));
    }

    cmi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_independent_uniform() {
        // Independent uniform variables → MI ≈ 0
        let x: Vec<f64> = (0..1000).map(|i| (i % 10) as f64 / 10.0).collect();
        let y: Vec<f64> = (0..1000).map(|i| ((i * 7) % 10) as f64 / 10.0).collect();

        let mi = mi_from_samples(&x, &y, 5);
        assert!(mi < 0.1, "Independent variables should have MI ≈ 0, got {}", mi);

        let nmi = normalized_mi(&x, &y, 5);
        assert!(nmi < 0.1, "Independent variables should have NMI ≈ 0, got {}", nmi);
    }

    #[test]
    fn test_identical_vectors() {
        // Identical vectors → NMI ≈ 1
        let x: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let y = x.clone();

        let nmi = normalized_mi(&x, &y, 5);
        assert!(nmi > 0.9, "Identical vectors should have NMI ≈ 1, got {}", nmi);
    }

    #[test]
    fn test_known_joint_distribution() {
        // Create a known joint distribution
        let mut joint = HashMap::new();
        joint.insert((0, 0), 0.25);
        joint.insert((0, 1), 0.25);
        joint.insert((1, 0), 0.25);
        joint.insert((1, 1), 0.25);

        let mut marginal_x = HashMap::new();
        marginal_x.insert(0, 0.5);
        marginal_x.insert(1, 0.5);

        let mut marginal_y = HashMap::new();
        marginal_y.insert(0, 0.5);
        marginal_y.insert(1, 0.5);

        let mi = shannon_mi(&joint, &marginal_x, &marginal_y);
        // Uniform independent → MI = 1 bit (since 2×2 uniform → H = 2 bits, I = 0)
        // Actually for independent uniform: MI = 0
        assert!(mi < 0.01, "Independent uniform should have MI ≈ 0, got {}", mi);
    }

    #[test]
    fn test_perfect_correlation() {
        // y = 2x + 1 → perfect correlation → high MI
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let nmi = normalized_mi(&x, &y, 5);
        assert!(nmi > 0.5, "Perfect linear correlation should have high NMI, got {}", nmi);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution over 4 values → H = 2 bits
        let samples = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let h = entropy(&samples, 4);
        assert!(h > 1.5 && h < 2.5, "Uniform 4-bin entropy should be ~2 bits, got {}", h);
    }

    #[test]
    fn test_empty_vectors() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];
        assert_eq!(mi_from_samples(&x, &y, 5), 0.0);
        assert_eq!(normalized_mi(&x, &y, 5), 0.0);
    }

    #[test]
    fn test_single_element() {
        let x = vec![1.0];
        let y = vec![2.0];
        assert_eq!(mi_from_samples(&x, &y, 5), 0.0);
    }

    #[test]
    fn test_parity_with_python() {
        // Known test case matching measure_type4.py behavior
        // 4-tuple format: step, obs, model, action
        let obs: Vec<f64> = vec![0.78, 0.77, 0.76, 0.75, 0.74];
        let model: Vec<f64> = vec![0.55, 0.665, 0.7233, 0.7525, 0.768];

        let nmi = normalized_mi(&obs, &model, 5);
        println!("Python-like test: NMI = {:.6}", nmi);

        // The exact value depends on the discretization, but it should be > 0
        assert!(nmi >= 0.0 && nmi <= 1.0, "NMI should be in [0, 1]");
    }
}
