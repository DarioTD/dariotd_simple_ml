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
struct Model {
    n: [Neuron2in; 3],
}

impl Model {
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
        Self { n }
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
        Self { n }
    }

    fn apply_diff(&mut self, g: Self, rate: f64) {
        for i in 0..g.n.len() {
            self.n[i].w1 -= rate * g.n[i].w1;
            self.n[i].w2 -= rate * g.n[i].w2;
            self.n[i].b -= rate * g.n[i].b;
        }
    }

    fn cost(&self) -> f64 {
        TRAINING_SET.iter().fold(0.0, |acc, i| {
            let x1 = i[0];
            let x2 = i[1];
            let y = self.forward(x1, x2);
            let d = y - i[2];
            acc + f64::abs(d * d * d)
        }) / TRAINING_SET.len() as f64
    }

    fn finite_diff(&mut self, eps: f64) -> Self {
        let mut g = Model::new();

        for i in 0..self.n.len() {
            let c = self.cost();

            let original_val = self.n[i].w1;
            self.n[i].w1 += eps;
            g.n[i].w1 = (self.cost() - c) / eps;
            self.n[i].w1 = original_val;

            let original_val = self.n[i].w2;
            self.n[i].w2 += eps;
            g.n[i].w2 = (self.cost() - c) / eps;
            self.n[i].w2 = original_val;

            let original_val = self.n[i].b;
            self.n[i].b += eps;
            g.n[i].b = (self.cost() - c) / eps;
            self.n[i].b = original_val;
        }
        g
    }

    fn forward(&self, x1: f64, x2: f64) -> f64 {
        let a = ATC_M.activate(x1 * self.n[0].w1 + x2 * self.n[0].w2 + self.n[0].b);
        let b = ATC_M.activate(x1 * self.n[1].w1 + x2 * self.n[1].w2 + self.n[1].b);
        ATC_M.activate(a * self.n[2].w1 + b * self.n[2].w2 + self.n[2].b)
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

fn main() {
    let mut m = Model::new_rand();
    let c = m.cost();
    println!("Original = m: {:#?}", m);
    println!("cost: {c}");
    println!();
    let rate = 1e-2;
    let eps = 1e-3;

    for _ in 0..1_000_000 {
        let g = m.finite_diff(eps);
        m.apply_diff(g, rate);
        //println!("{i} = m: {:#?}", m);
        //println!("cost: {}", m.cost());
        //println!();
    }
    println!("-------------------------------");
    println!("activation method: {:?}", ATC_M);
    println!();
    println!("Final = m: {:#?}", m);
    println!("cost: {}", m.cost());
    println!();
    for i in TRAINING_SET {
        let x1 = i[0];
        let x2 = i[1];
        let ex_out = i[2];
        let y = m.forward(x1, x2);
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
