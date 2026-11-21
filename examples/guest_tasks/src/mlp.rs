//! A small multi-layer perceptron (input → hidden ReLU → softmax classes), written
//! without dependencies so it runs inside a wasm task.
//!
//! Parameters are one flat f32 vector `[W1 | b1 | W2 | b2]`:
//! `W1` is `hidden × input` (row per hidden unit), `W2` is `classes × hidden`.
//! Inputs are u8 pixels scaled to `0..1`.

#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub input: usize,
    pub hidden: usize,
    pub classes: usize,
}

impl Dims {
    /// From the task's `config` buffer: `[input, hidden, classes]`.
    pub fn from_config(config: &[u32]) -> Self {
        assert!(config.len() >= 3, "config must be [input, hidden, classes]");
        Self { input: config[0] as usize, hidden: config[1] as usize, classes: config[2] as usize }
    }

    fn w1(&self) -> usize {
        0
    }
    fn b1(&self) -> usize {
        self.hidden * self.input
    }
    fn w2(&self) -> usize {
        self.b1() + self.hidden
    }
    fn b2(&self) -> usize {
        self.w2() + self.classes * self.hidden
    }
    pub fn param_count(&self) -> usize {
        self.b2() + self.classes
    }
}

/// Per-thread working memory, reused across samples.
pub struct Scratch {
    /// Non-zero inputs as (index, value): MNIST images are ~80% zeros.
    nz: Vec<(u32, f32)>,
    z1: Vec<f32>,
    a1: Vec<f32>,
    probs: Vec<f32>,
    dz1: Vec<f32>,
}

impl Scratch {
    pub fn new(d: Dims) -> Self {
        Self {
            nz: Vec::with_capacity(d.input),
            z1: vec![0.0; d.hidden],
            a1: vec![0.0; d.hidden],
            probs: vec![0.0; d.classes],
            dz1: vec![0.0; d.hidden],
        }
    }
}

/// Forward pass; leaves softmax probabilities in `s.probs`.
fn forward(d: Dims, params: &[f32], pixels: &[u8], s: &mut Scratch) {
    s.nz.clear();
    for (k, &p) in pixels.iter().enumerate() {
        if p != 0 {
            s.nz.push((k as u32, p as f32 / 255.0));
        }
    }

    let w1 = &params[d.w1()..d.b1()];
    let b1 = &params[d.b1()..d.w2()];
    for j in 0..d.hidden {
        let row = &w1[j * d.input..(j + 1) * d.input];
        let mut z = b1[j];
        for &(k, x) in &s.nz {
            z += row[k as usize] * x;
        }
        s.z1[j] = z;
        s.a1[j] = z.max(0.0);
    }

    let w2 = &params[d.w2()..d.b2()];
    let b2 = &params[d.b2()..d.param_count()];
    let mut max = f32::NEG_INFINITY;
    for o in 0..d.classes {
        let row = &w2[o * d.hidden..(o + 1) * d.hidden];
        let z = b2[o] + row.iter().zip(&s.a1).map(|(w, a)| w * a).sum::<f32>();
        s.probs[o] = z;
        max = max.max(z);
    }
    let mut total = 0.0;
    for p in s.probs.iter_mut() {
        *p = (*p - max).exp();
        total += *p;
    }
    for p in s.probs.iter_mut() {
        *p /= total;
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// Predicted class for one input.
pub fn predict(d: Dims, params: &[f32], pixels: &[u8], s: &mut Scratch) -> u32 {
    forward(d, params, pixels, s);
    argmax(&s.probs) as u32
}

/// Adds this sample's cross-entropy gradient to `grads` (same layout as `params`).
/// Returns `(loss, correct)`.
pub fn accumulate_grad(
    d: Dims,
    params: &[f32],
    pixels: &[u8],
    label: u32,
    grads: &mut [f32],
    s: &mut Scratch,
) -> (f32, bool) {
    forward(d, params, pixels, s);
    let label = label as usize;
    let loss = -s.probs[label].max(1e-12).ln();
    let correct = argmax(&s.probs) == label;

    // dL/dz2 = probs - onehot(label)
    s.probs[label] -= 1.0;

    let w2 = d.w2();
    let b2 = d.b2();
    s.dz1.iter_mut().for_each(|v| *v = 0.0);
    for o in 0..d.classes {
        let dz2 = s.probs[o];
        grads[b2 + o] += dz2;
        let row = w2 + o * d.hidden;
        for j in 0..d.hidden {
            grads[row + j] += dz2 * s.a1[j];
            s.dz1[j] += params[row + j] * dz2;
        }
    }

    let b1 = d.b1();
    for j in 0..d.hidden {
        if s.z1[j] <= 0.0 {
            continue; // ReLU blocks the gradient
        }
        let dz = s.dz1[j];
        grads[b1 + j] += dz;
        let row = &mut grads[j * d.input..(j + 1) * d.input];
        for &(k, x) in &s.nz {
            row[k as usize] += dz * x;
        }
    }

    (loss, correct)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loss(d: Dims, params: &[f32], pixels: &[u8], label: u32) -> f32 {
        let mut s = Scratch::new(d);
        forward(d, params, pixels, &mut s);
        -s.probs[label as usize].ln()
    }

    /// Compares the analytic gradient with central finite differences.
    #[test]
    fn gradient_matches_finite_differences() {
        let d = Dims { input: 7, hidden: 5, classes: 3 };
        let mut seed = 0x2545_f491u32;
        let mut rand = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) - 0.5
        };
        let params: Vec<f32> = (0..d.param_count()).map(|_| rand()).collect();
        let pixels = [0u8, 200, 13, 0, 255, 90, 47];
        let label = 2;

        let mut grads = vec![0.0; d.param_count()];
        let mut s = Scratch::new(d);
        accumulate_grad(d, &params, &pixels, label, &mut grads, &mut s);

        let eps = 1e-2;
        for i in 0..d.param_count() {
            let mut plus = params.clone();
            plus[i] += eps;
            let mut minus = params.clone();
            minus[i] -= eps;
            let numeric = (loss(d, &plus, &pixels, label) - loss(d, &minus, &pixels, label)) / (2.0 * eps);
            let diff = (numeric - grads[i]).abs();
            assert!(
                diff < 2e-3 + 2e-2 * numeric.abs(),
                "param {i}: analytic {} vs numeric {numeric}",
                grads[i]
            );
        }
    }

    #[test]
    fn learns_a_toy_problem() {
        // Class = which of the 3 inputs is lit.
        let d = Dims { input: 3, hidden: 8, classes: 3 };
        let mut params: Vec<f32> = (0..d.param_count()).map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.05).collect();
        let samples = [([255u8, 0, 0], 0u32), ([0, 255, 0], 1), ([0, 0, 255], 2)];
        let mut s = Scratch::new(d);
        for _ in 0..300 {
            let mut grads = vec![0.0; d.param_count()];
            for (x, y) in &samples {
                accumulate_grad(d, &params, x, *y, &mut grads, &mut s);
            }
            for (p, g) in params.iter_mut().zip(&grads) {
                *p -= 0.5 * g;
            }
        }
        for (x, y) in &samples {
            assert_eq!(predict(d, &params, x, &mut s), *y);
        }
    }
}
