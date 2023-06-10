use rand::Rng;

const TRAINING_SET: [(f64, f64); 5] = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0)];

fn cost(weight: f64, bias: f64) -> f64 {
    let mut result = 0.0;
    for i in 0..TRAINING_SET.len() {
        let input = TRAINING_SET[i].0;
        let expected_output = TRAINING_SET[i].1;
        let output = input * weight + bias;
        let distance = output - expected_output;
        result += f64::abs(distance * distance * distance);
    }
    result /= TRAINING_SET.len() as f64;
    result
}

fn main() {
    // output = input * weight `weight` 1 parameter
    let mut weight = rand::thread_rng().gen_range(0.0..=20.0);
    let mut bias = rand::thread_rng().gen_range(0.0..=10.0);
    let weight1 = weight;
    let bias1 = bias;
    let rate = 1e-2;
    let eps = 1e-3;

    for _ in 0..1_000_000 {
        let c = cost(weight, bias);
        let d_weight = (cost(weight + eps, bias) - c) / weight;
        let d_bias = (cost(weight, bias + eps) - c) / weight;
        weight -= rate * d_weight;
        bias -= rate * d_bias;
    }
    println!();
    println!("Original = weight: {}, bias: {}", weight1, bias1);
    println!(
        "Final = weight: {}, bias: {}, cost: {}",
        weight,
        bias,
        cost(weight, bias)
    );
}
