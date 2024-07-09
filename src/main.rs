mod training_data_sets;

use dariotd_simple_ml::{Act, Model};
use training_data_sets::{training_data_set, TrainingDataSet};

fn main() {
    let training_data_set = training_data_set(&TrainingDataSet::Xor);
    let neurons_per_layer = vec![
        training_data_set[0].0.len(),
        4,
        training_data_set[0].1.len(),
    ];
    let act_m = Act::Sigmoid;
    let mut model = Model::build(&act_m, &training_data_set, &neurons_per_layer);
    let cost = model.cost();
    println!("-------------------------------");
    println!("activation method: {act_m:?}");
    println!();
    println!("Initial Model: {model:#?}");
    println!("Initial Cost: {cost}");
    println!();

    let rate = 1e-0;
    let eps = 1e-3;
    let n_iterations = 2 * 100_000;
    for i in 1..=n_iterations {
        let gradient = model.finite_diff(eps);
        let prev_cost = model.cost();
        model.apply_diff(&gradient, rate);
        let cost = model.cost();
        println!("iter: {i} -> {n_iterations}");
        println!("cost: {cost}");
        println!();
        if cost > prev_cost {
            break;
        }
    }

    println!("-------------------------------");
    println!("activation method: {act_m:?}");
    println!();
    println!("Final Model: {model:#?}");
    println!("Final Cost: {}", model.cost());
    println!();
    println!("-------------------------------");
    println!("overall:");
    for data_set in &training_data_set {
        let inputs = &data_set.0;
        let exp_outputs = &data_set.1;
        let outputs = model.forward(inputs);
        println!("-------------------------------");
        println!("inputs: {inputs:?}");
        println!("expected_outputs: {exp_outputs:?}");
        println!("outputs: {outputs:?}");
    }
}
