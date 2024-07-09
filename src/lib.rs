use rand::Rng;

#[allow(dead_code)]
#[derive(Debug)]
pub enum Act {
    None,
    Relu,
    Sigmoid,
    Sine,
    Tanh,
}

impl Act {
    pub fn apply_act(&self, x: f64) -> f64 {
        match self {
            Self::None => x,
            Self::Relu => f64::max(0.0, x),
            Self::Sigmoid => 1.0 / (1.0 + f64::exp(-x)),
            Self::Sine => f64::sin(x),
            Self::Tanh => f64::tanh(x),
        }
    }
}

#[derive(Debug)]
pub struct Neuron {
    pub id: usize,
    pub bias: f64,
    pub weights: Vec<f64>,
}

#[derive(Debug)]
pub struct Layer {
    pub id: usize,
    pub neurons: Vec<Neuron>,
}

#[derive(Debug)]
pub struct Model<'a> {
    pub act_mode: &'a Act,
    pub training_data_set: &'a [(Vec<f64>, Vec<f64>)],
    pub layers: Vec<Layer>,
}

impl Layer {
    fn build(id: usize, n_neurons: usize, n_weights: usize) -> Self {
        let mut neurons = Vec::with_capacity(n_neurons);
        (0..n_neurons).for_each(|i| {
            let mut weights = Vec::with_capacity(n_weights);
            (0..n_weights).for_each(|_| {
                weights.push(rand::thread_rng().gen_range(0.0..=4.0) - 2.0);
            });
            neurons.push(Neuron {
                id: i,
                bias: rand::thread_rng().gen_range(0.0..=4.0) - 2.0,
                weights,
            });
        });
        Self { id, neurons }
    }

    fn build_with_zeros(id: usize, n_neurons: usize, n_weights: usize) -> Self {
        let mut neurons = Vec::with_capacity(n_neurons);
        (0..n_neurons).for_each(|i| {
            let mut weights = Vec::with_capacity(n_weights);
            (0..n_weights).for_each(|_| {
                weights.push(0.0);
            });
            neurons.push(Neuron {
                id: i,
                bias: 0.0,
                weights,
            });
        });
        Self { id, neurons }
    }
}

impl<'a> Model<'a> {
    pub fn apply_diff(&mut self, gradient: &Self, learning_rate: f64) {
        self.layers
            .iter_mut()
            .zip(&gradient.layers)
            .for_each(|(layer, g_layer)| {
                layer
                    .neurons
                    .iter_mut()
                    .zip(&g_layer.neurons)
                    .for_each(|(neuron, g_neuron)| {
                        neuron.bias -= learning_rate * g_neuron.bias;
                        neuron.weights.iter_mut().zip(&g_neuron.weights).for_each(
                            |(weight, g_weight)| {
                                *weight -= learning_rate * g_weight;
                            },
                        );
                    });
            });
    }

    pub fn build(
        act_mode: &'a Act,
        training_data_set: &'a [(Vec<f64>, Vec<f64>)],
        n_neuron_per_layer: &[usize],
    ) -> Self {
        let mut n_layers = Vec::new();
        let mut prev_n_neuron = vec![n_neuron_per_layer[0]];
        n_neuron_per_layer
            .iter()
            .enumerate()
            .for_each(|(id, &n_neuron)| {
                n_layers.push((n_neuron, prev_n_neuron[id]));
                prev_n_neuron.push(n_neuron);
            });
        let layers = n_layers
            .iter()
            .enumerate()
            .map(|(id, &(n_neurons, n_weights))| Layer::build(id, n_neurons, n_weights))
            .collect::<Vec<_>>();
        Self {
            act_mode,
            training_data_set,
            layers,
        }
    }

    pub fn build_with_zeros(
        act_mode: &'a Act,
        training_data_set: &'a [(Vec<f64>, Vec<f64>)],
        n_neuron_per_layer: &[usize],
    ) -> Self {
        let mut n_layers = Vec::new();
        let mut prev_n_neuron = vec![n_neuron_per_layer[0]];
        n_neuron_per_layer
            .iter()
            .enumerate()
            .for_each(|(id, &n_neuron)| {
                n_layers.push((n_neuron, prev_n_neuron[id]));
                prev_n_neuron.push(n_neuron);
            });
        let layers = n_layers
            .iter()
            .enumerate()
            .map(|(id, &(n_neurons, n_weights))| Layer::build_with_zeros(id, n_neurons, n_weights))
            .collect::<Vec<_>>();
        Self {
            act_mode,
            training_data_set,
            layers,
        }
    }

    pub fn cost(&self) -> f64 {
        self.training_data_set
            .iter()
            .map(|(t_inputs, t_outputs)| {
                let outputs = self.forward(t_inputs);
                outputs
                    .iter()
                    .zip(t_outputs)
                    .fold(0.0, |acc, (output, expected_output)| {
                        let dis = output - expected_output;
                        acc + dis * dis
                    })
                    / (outputs.len() as f64)
            })
            .sum::<f64>()
            / (self.training_data_set.len() as f64)
    }

    pub fn finite_diff(&mut self, eps: f64) -> Self {
        let mut n_neuron_per_layer = Vec::new();
        self.layers.iter().for_each(|layer| {
            n_neuron_per_layer.push(layer.neurons.len());
        });
        let mut gradient =
            Self::build_with_zeros(self.act_mode, self.training_data_set, &n_neuron_per_layer);

        for i in 0..gradient.layers.len() {
            for j in 0..gradient.layers[i].neurons.len() {
                let cost = self.cost();

                let original_val = self.layers[i].neurons[j].bias;
                self.layers[i].neurons[j].bias += eps;
                gradient.layers[i].neurons[j].bias = (self.cost() - cost) / eps;
                self.layers[i].neurons[j].bias = original_val;

                for k in 0..self.layers[i].neurons[j].weights.len() {
                    let original_val = self.layers[i].neurons[j].weights[k];
                    self.layers[i].neurons[j].weights[k] += eps;
                    gradient.layers[i].neurons[j].weights[k] = (self.cost() - cost) / eps;
                    self.layers[i].neurons[j].weights[k] = original_val;
                }
            }
        }
        gradient
    }

    pub fn forward(&self, inputs: &[f64]) -> Vec<f64> {
        assert_eq!(self.layers[0].neurons[0].weights.len(), inputs.len());
        let mut act_layers = Vec::new();
        act_layers.push(
            self.layers[0]
                .neurons
                .iter()
                .map(|neuron| {
                    Act::apply_act(
                        self.act_mode,
                        neuron
                            .weights
                            .iter()
                            .zip(inputs)
                            .map(|(weight, input)| weight * input)
                            .sum::<f64>()
                            + neuron.bias,
                    )
                })
                .collect::<Vec<_>>(),
        );
        (1..self.layers.len()).for_each(|i| {
            act_layers.push(
                self.layers[i]
                    .neurons
                    .iter()
                    .map(|neuron| {
                        Act::apply_act(
                            self.act_mode,
                            neuron
                                .weights
                                .iter()
                                .zip(act_layers.last().unwrap())
                                .map(|(weight, input)| weight * input)
                                .sum::<f64>()
                                + neuron.bias,
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        });
        act_layers.last().unwrap().clone()
    }
}
