mod training_sets;

pub const TRAINING_SET: [[f64; 3]; 4] = training_sets::_XOR;
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
    pub fn apply_act(&self, x: f64) -> f64 {
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
pub struct Neuron {
    pub bias: f64,
    pub weights: Vec<f64>,
}

#[derive(Debug)]
pub struct Model {
    pub neurons: Vec<Neuron>,
}

impl Model {
    pub fn build(n_neurons: usize) -> Self {
        let mut neurons = Vec::with_capacity(n_neurons);
        for _ in 0..n_neurons {
            neurons.push(Neuron {
                bias: rand::random(),
                weights: vec![rand::random(), rand::random()],
            });
        }
        Self { neurons }
    }

    fn build_zeros(n_neurons: usize) -> Self {
        let mut neurons = Vec::with_capacity(n_neurons);
        for _ in 0..n_neurons {
            neurons.push(Neuron {
                bias: 0.0,
                weights: vec![0.0, 0.0],
            });
        }
        Self { neurons }
    }

    pub fn apply_diff(&mut self, gradient: &Self, rate: f64) {
        let n_neurons = self.neurons.len();
        for i in 0..n_neurons {
            self.neurons[i].bias -= rate * gradient.neurons[i].bias;
            for j in 0..gradient.neurons[i].weights.len() {
                self.neurons[i].weights[j] -= rate * gradient.neurons[i].weights[j];
            }
        }
    }

    pub fn cost(&self, training_set: &[[f64; 3]; 4]) -> f64 {
        training_set.iter().fold(0.0, |acc, i| {
            let input = vec![i[0], i[1]];
            let output = self.forward(&input);
            let dis = output - i[2];
            acc + f64::abs(dis * dis * dis)
        }) / training_set.len() as f64
    }

    pub fn finite_diff(&mut self, eps: f64) -> Self {
        let n_neurons = self.neurons.len();
        let mut gradient = Self::build_zeros(n_neurons);

        for i in 0..n_neurons {
            let cost = self.cost(&TRAINING_SET);

            let original_val = self.neurons[i].bias;
            self.neurons[i].bias += eps;
            gradient.neurons[i].bias = (self.cost(&TRAINING_SET) - cost) / eps;
            self.neurons[i].bias = original_val;

            for j in 0..self.neurons[i].weights.len() {
                let original_val = self.neurons[i].weights[j];
                self.neurons[i].weights[j] += eps;
                gradient.neurons[i].weights[j] = (self.cost(&TRAINING_SET) - cost) / eps;
                self.neurons[i].weights[j] = original_val;
            }
        }
        gradient
    }

    pub fn forward(&self, input: &[f64]) -> f64 {
        let a = ATC_M.apply_act(
            self.neurons[0].bias
                + input[0] * self.neurons[0].weights[0]
                + input[1] * self.neurons[0].weights[1],
        );
        let b = ATC_M.apply_act(
            self.neurons[1].bias
                + input[0] * self.neurons[1].weights[0]
                + input[1] * self.neurons[1].weights[1],
        );
        ATC_M.apply_act(
            self.neurons[2].bias + a * self.neurons[2].weights[0] + b * self.neurons[2].weights[1],
        )
    }
}
