use rand::Rng;

#[derive(Clone)]
struct PatientRecord {
    features: Vec<f64>,
    outcome: f64, // Using f64 for easier math during gradient descent
}

struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
}

impl LogisticRegression {
    fn new(num_features: usize) -> Self {
        Self {
            weights: vec![0.0; num_features],
            bias: 0.0,
        }
    }

    fn sigmoid(z: f64) -> f64 {
        // Clamp to prevent overflow
        let z_clamped = z.clamp(-250.0, 250.0);
        1.0 / (1.0 + (-z_clamped).exp())
    }

    fn predict_prob(&self, features: &[f64]) -> f64 {
        let mut z = self.bias;
        for (w, x) in self.weights.iter().zip(features.iter()) {
            z += w * x;
        }
        Self::sigmoid(z)
    }

    fn train(&mut self, x: &[Vec<f64>], y: &[f64], lr: f64, epochs: usize) {
        let m = x.len() as f64;
        let num_features = self.weights.len();

        for _ in 0..epochs {
            let mut bias_grad = 0.0;
            let mut weight_grads = vec![0.0; num_features];

            for i in 0..x.len() {
                let pred = self.predict_prob(&x[i]);
                let error = pred - y[i];

                bias_grad += error;
                for j in 0..num_features {
                    weight_grads[j] += error * x[i][j];
                }
            }

            self.bias -= lr * (bias_grad / m);
            for j in 0..num_features {
                self.weights[j] -= lr * (weight_grads[j] / m);
            }
        }
    }
}

// Generates synthetic deep phenotype data
fn generate_data(count: usize) -> Vec<PatientRecord> {
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(count);

    for _ in 0..count {
        let age = rng.gen_range(30.0..80.0);
        let glucose = rng.gen_range(90.0..140.0);
        let comorbidity = rng.gen_range(0.1..1.0);
        let bp = rng.gen_range(110.0..150.0);

        // Underlying generation rule
        let risk_score = age * 0.02 + glucose * 0.01 + comorbidity * 2.0 + bp * 0.01 + rng.gen_range(0.0..0.5);
        let outcome = if risk_score > 3.5 { 1.0 } else { 0.0 };

        data.push(PatientRecord {
            features: vec![age, glucose, comorbidity, bp],
            outcome,
        });
    }
    data
}

// Z-Score Standardization (Fit on train, transform train & test to avoid leakage)
fn standardize(train: &[PatientRecord], test: &[PatientRecord]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let num_features = train[0].features.len();
    let n = train.len() as f64;
    
    let mut means = vec![0.0; num_features];
    let mut stdevs = vec![0.0; num_features];

    // Calculate Means
    for p in train {
        for i in 0..num_features {
            means[i] += p.features[i];
        }
    }
    for m in &mut means { *m /= n; }

    // Calculate Standard Deviations
    for p in train {
        for i in 0..num_features {
            stdevs[i] += (p.features[i] - means[i]).powi(2);
        }
    }
    for s in &mut stdevs { 
        *s = (*s / n).sqrt();
        if *s == 0.0 { *s = 1.0; } // Prevent division by zero
    }

    // Helper closure to scale data
    let scale = |data: &[PatientRecord]| -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut scaled_x = Vec::with_capacity(data.len());
        let mut y = Vec::with_capacity(data.len());
        for p in data {
            let mut scaled_features = Vec::with_capacity(num_features);
            for i in 0..num_features {
                scaled_features.push((p.features[i] - means[i]) / stdevs[i]);
            }
            scaled_x.push(scaled_features);
            y.push(p.outcome);
        }
        (scaled_x, y)
    };

    let (train_x, train_y) = scale(train);
    let (test_x, test_y) = scale(test);
    
    (train_x, train_y, test_x, test_y)
}

fn main() {
    println!("=== Rust High-Performance ML Pipeline: Clinical Risk Prediction ===");
    
    let count = 500;
    let k_folds = 5;
    let epochs = 1000;
    let learning_rate = 0.1;

    let mut dataset = generate_data(count);
    
    // Quick pseudo-shuffle by swapping random indices
    let mut rng = rand::thread_rng();
    for i in 0..dataset.len() {
        let swap_idx = rng.gen_range(0..dataset.len());
        dataset.swap(i, swap_idx);
    }

    let fold_size = dataset.len() / k_folds;
    let mut total_accuracy = 0.0;

    println!("Running {}-Fold Cross Validation with Batch Gradient Descent...\n", k_folds);

    for k in 0..k_folds {
        let test_start = k * fold_size;
        let test_end = test_start + fold_size;

        let test_data = &dataset[test_start..test_end];
        let mut train_data = Vec::new();
        train_data.extend_from_slice(&dataset[..test_start]);
        train_data.extend_from_slice(&dataset[test_end..]);

        let (train_x, train_y, test_x, test_y) = standardize(&train_data, test_data);

        let mut model = LogisticRegression::new(4);
        model.train(&train_x, &train_y, learning_rate, epochs);

        let mut correct = 0;
        for i in 0..test_x.len() {
            let pred_prob = model.predict_prob(&test_x[i]);
            let pred_class = if pred_prob >= 0.5 { 1.0 } else { 0.0 };
            
            if (pred_class - test_y[i]).abs() < f64::EPSILON {
                correct += 1;
            }
        }

        let accuracy = correct as f64 / test_data.len() as f64;
        total_accuracy += accuracy;
        println!("Fold {} Accuracy: {:.2}%", k + 1, accuracy * 100.0);
    }

    println!("--------------------------------------------------");
    println!("Average Model Accuracy: {:.2}%", (total_accuracy / k_folds as f64) * 100.0);
    println!("Pipeline Complete.");
}
