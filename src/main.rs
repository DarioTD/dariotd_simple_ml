use rand::Rng;

const TRAINING_SET: [(f64, f64, f64); 4] = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
];

fn cost(w1: f64, w2: f64) -> f64 {
    let mut result = 0.0;
    for i in 0..TRAINING_SET.len() {
        let x1 = TRAINING_SET[i].0;
        let x2 = TRAINING_SET[i].1;
        let y = x1 * w1 + x2 * w2;
        let d = y - TRAINING_SET[i].2;
        result += f64::abs(d * d * d);
    }
    result /= TRAINING_SET.len() as f64;
    result
}

fn main() {
    let mut w1 = rand::thread_rng().gen_range(0.0..=10.0);
    let mut w2 = rand::thread_rng().gen_range(0.0..=10.0);
    let o_w1 = w1;
    let o_w2 = w2;
    let rate = 1e-2;
    let eps = 1e-2;

    for i in 0..10_000 {
        let c = cost(w1, w2);
        let d_w1 = (cost(w1 + eps, w2) - c) / w1;
        let d_w2 = (cost(w1, w2 + eps) - c) / w2;
        w1 -= rate * d_w1;
        w2 -= rate * d_w2;
        println!("{i} = w1: {}, w2: {}, c: {}", w1, w2, cost(w1, w2));
    }
    println!("-------------------------------");
    println!();
    println!("Original = w1: {}, w2: {}", o_w1, o_w2);
    println!("Final = w1: {}, w2: {}, c: {}", w1, w2, cost(w1, w2));
    println!();
    for i in 0..TRAINING_SET.len() {
        let x1 = TRAINING_SET[i].0;
        let x2 = TRAINING_SET[i].1;
        let ex_out = TRAINING_SET[i].2;
        let y = x1 * w1 + x2 * w2;
        println!("x1: {}, x2: {}, ex_out: {}, y: {}", x1, x2, ex_out, y);
    }
}
