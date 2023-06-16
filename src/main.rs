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
                w1: 0.0,
                w2: 0.0,
                b: 0.0,
            },
            Neuron2in {
                w1: 0.0,
                w2: 0.0,
                b: 0.0,
            },
            Neuron2in {
                w1: 0.0,
                w2: 0.0,
                b: 0.0,
            },
        ];
        Xor { n }
    }
    fn new_rand() -> Self {
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
const _OR_TRAINING_SET: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];
const _NOR_TRAINING_SET: [[f64; 3]; 4] = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
];
const _XOR_TRAINING_SET: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
];
const _NXOR_TRAINING_SET: [[f64; 3]; 4] = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
];
const TRAINING_SET: &[[f64; 3]; 4] = &_XOR_TRAINING_SET;
const ATC_M: Act = Act::Sigmoid;

fn apply_diff(m: &mut Xor, g: Xor, rate: f64) {
    for i in 0..g.n.len() {
        m.n[i].w1 -= rate * g.n[i].w1;
        m.n[i].w2 -= rate * g.n[i].w2;
        m.n[i].b -= rate * g.n[i].b;
    }
}

fn cost(m: &Xor) -> f64 {
    TRAINING_SET.iter().fold(0.0, |acc, i| {
        let x1 = i[0];
        let x2 = i[1];
        let y = forwarding(m, x1, x2);
        let d = y - i[2];
        acc + f64::abs(d * d * d)
    }) / TRAINING_SET.len() as f64
}

fn finite_diff(m: &mut Xor, eps: f64) -> Xor {
    let mut g = Xor::new();

    for i in 0..m.n.len() {
        let c = cost(m);

        let original = m.n[i].w1;
        m.n[i].w1 += eps;
        g.n[i].w1 = (cost(m) - c) / eps;
        m.n[i].w1 = original;

        let original = m.n[i].w2;
        m.n[i].w2 += eps;
        g.n[i].w2 = (cost(m) - c) / eps;
        m.n[i].w2 = original;

        let original = m.n[i].b;
        m.n[i].b += eps;
        g.n[i].b = (cost(m) - c) / eps;
        m.n[i].b = original;
    }
    g
}

fn forwarding(m: &Xor, x: f64, y: f64) -> f64 {
    let a = ATC_M.activate(x * m.n[0].w1 + y * m.n[0].w2 + m.n[0].b);
    let b = ATC_M.activate(x * m.n[1].w1 + y * m.n[1].w2 + m.n[1].b);
    ATC_M.activate(a * m.n[2].w1 + b * m.n[2].w2 + m.n[2].b)
}

fn main() {
    let mut m = Xor::new_rand();
    let c = cost(&m);
    println!("Original = m: {:#?}", m);
    println!("cost: {c}");
    println!();
    let rate = 1e-2;
    let eps = 1e-3;

    for _ in 0..1_000_000 {
        let g = finite_diff(&mut m, eps);
        apply_diff(&mut m, g, rate);
        //println!("{i} = m: {:#?}", m);
        //println!("cost: {}", cost(&m));
        //println!();
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
    println!("-------------------------------");
    println!("a:");
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{} | {} = {}",
                i,
                j,
                ATC_M.activate(i as f64 * m.n[0].w1 + j as f64 * m.n[0].w2 + m.n[0].b)
            );
        }
    }
    println!("b:");
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{} | {} = {}",
                i,
                j,
                ATC_M.activate(i as f64 * m.n[1].w1 + j as f64 * m.n[1].w2 + m.n[1].b)
            );
        }
    }
    println!("c:");
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{} | {} = {}",
                i,
                j,
                ATC_M.activate(i as f64 * m.n[2].w1 + j as f64 * m.n[2].w2 + m.n[2].b)
            );
        }
    }
}
