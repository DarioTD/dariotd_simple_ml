use rand::Rng;

#[allow(dead_code)]
#[derive(Debug)]
enum Act {
    None,
    Sigmoid,
    Sine,
    Tanh,
}

impl Act {
    fn activate(&self, x: f64) -> f64 {
        match self {
            Act::None => x,
            Act::Sigmoid => Act::sigmoid(x),
            Act::Sine => x.sin(),
            Act::Tanh => x.tanh(),
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-x))
    }
}

#[derive(Debug)]
struct Neuron2in {
    w1: f64,
    w2: f64,
    b: f64,
}

#[derive(Debug)]
struct Xor {
    n: [Neuron2in; 3],
}

impl Xor {
    fn new() -> Self {
        let n = [
            Neuron2in {
                w1: rand::thread_rng().gen_range(0.0..=1.0),
                w2: rand::thread_rng().gen_range(0.0..=1.0),
                b: rand::thread_rng().gen_range(0.0..=1.0),
            },
            Neuron2in {
                w1: rand::thread_rng().gen_range(0.0..=1.0),
                w2: rand::thread_rng().gen_range(0.0..=1.0),
                b: rand::thread_rng().gen_range(0.0..=1.0),
            },
            Neuron2in {
                w1: rand::thread_rng().gen_range(0.0..=1.0),
                w2: rand::thread_rng().gen_range(0.0..=1.0),
                b: rand::thread_rng().gen_range(0.0..=1.0),
            },
        ];
        Xor { n }
    }
}

const _XOR_TRAINING_SET: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
];
const _OR_TRAINING_SET: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];
const _AND_TRAINING_SET: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
];
const _NAND_TRAINING_SET: [[f64; 3]; 4] = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
];
const TRAINING_SET: &[[f64; 3]; 4] = &_XOR_TRAINING_SET;
const ATC_M: Act = Act::Sigmoid;

fn forwarding(m: &Xor, x: f64, y: f64) -> f64 {
    let a = ATC_M.activate(x * m.n[0].w1 + y * m.n[0].w2 + m.n[0].b);
    let b = ATC_M.activate(x * m.n[1].w1 + y * m.n[1].w2 + m.n[1].b);
    ATC_M.activate(a * m.n[2].w1 + b * m.n[2].w2 + m.n[2].b)
}

fn cost(m: &Xor) -> f64 {
    TRAINING_SET.iter().fold(0.0, |acc, i| {
        let x1 = i[0];
        let x2 = i[1];
        let y = forwarding(&m, x1, x2);
        let d = y - i[2];
        acc + f64::abs(d * d * d)
    }) / TRAINING_SET.len() as f64
}

fn main() {
    let mut m = Xor::new();
    let c = cost(&mut m);
    println!("Original = m: {:#?}", m);
    println!("cost: {c}");
    println!();
    let rate = 1e-2;
    let eps = 1e-3;

    for i in 0..500_000 {
        for j in 0..3 {
            let c = cost(&m);

            m.n[j].w1 += eps;
            let d_w1 = (cost(&m) - c) / eps;
            m.n[j].w1 -= eps;

            m.n[j].w2 += eps;
            let d_w2 = (cost(&m) - c) / eps;
            m.n[j].w2 -= eps;

            m.n[j].b += eps;
            let d_b = (cost(&m) - c) / eps;
            m.n[j].b -= eps;

            m.n[j].w1 -= rate * d_w1;
            m.n[j].w2 -= rate * d_w2;
            m.n[j].b -= rate * d_b;
        }
        println!("{i} = m: {:#?}", m);
        println!("cost: {}", cost(&m));
        println!();
    }
    println!("-------------------------------");
    println!("activation method: {:?}", ATC_M);
    println!();
    println!("Final = m: {:#?}", m);
    println!("cost: {}", cost(&m));
    println!();
    for i in TRAINING_SET {
        let x1 = i[0];
        let x2 = i[1];
        let ex_out = i[2];
        let y = forwarding(&m, x1, x2);
        println!("x1: {}, x2: {}, ex_out: {}, y: {}", x1, x2, ex_out, y);
    }
}
