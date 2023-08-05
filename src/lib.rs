use rand::Rng;

pub mod training_set;

pub const TRAINING_SET: &[[f64; 3]; 4] = &training_set::_XOR_TRAINING_SET;
pub const ATC_M: Act = Act::Sigmoid;

#[allow(dead_code)]
#[derive(Debug)]
pub enum Act {
    None,
    Sigmoid,
    Sine,
    Tanh,
}

impl Act {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            Self::None => x,
            Self::Sigmoid => Self::sigmoid(x),
            Self::Sine => x.sin(),
            Self::Tanh => x.tanh(),
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-x))
    }
}

#[derive(Debug)]
pub struct Neuron2in {
    pub w1: f64,
    pub w2: f64,
    pub b: f64,
}

#[derive(Debug)]
pub struct Model {
    pub n: [Neuron2in; 3],
}

impl Model {
    pub const fn new() -> Self {
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

    pub fn new_rand() -> Self {
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

    pub fn apply_diff(&mut self, g: &Self, rate: f64) {
        for i in 0..g.n.len() {
            self.n[i].w1 -= rate * g.n[i].w1;
            self.n[i].w2 -= rate * g.n[i].w2;
            self.n[i].b -= rate * g.n[i].b;
        }
    }

    pub fn cost(&self) -> f64 {
        TRAINING_SET.iter().fold(0.0, |acc, i| {
            let x1 = i[0];
            let x2 = i[1];
            let y = self.forward(x1, x2);
            let d = y - i[2];
            acc + f64::abs(d * d * d)
        }) / TRAINING_SET.len() as f64
    }

    pub fn finite_diff(&mut self, eps: f64) -> Self {
        let mut g = Self::new();

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

    pub fn forward(&self, x1: f64, x2: f64) -> f64 {
        let a = ATC_M.activate(x1.mul_add(self.n[0].w1, x2 * self.n[0].w2) + self.n[0].b);
        let b = ATC_M.activate(x1.mul_add(self.n[1].w1, x2 * self.n[1].w2) + self.n[1].b);
        ATC_M.activate(a.mul_add(self.n[2].w1, b * self.n[2].w2) + self.n[2].b)
    }
}
