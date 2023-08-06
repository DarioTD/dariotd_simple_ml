use dariotd_simple_ml::{Model, ATC_M, TRAINING_SET};

fn main() {
    let n_neurons = 3;
    let mut model = Model::build(n_neurons);
    let cost = model.cost(&TRAINING_SET);
    println!("Original: {model:#?}");
    println!("cost: {cost}");
    println!();
    let rate = 1e-1;
    let eps = 1e-3;
    let n_iterations = 100_000;

    for _ in 0..n_iterations {
        let gradient = model.finite_diff(eps);
        model.apply_diff(&gradient, rate);
        //println!("{i}: {model:#?}");
        //println!("cost: {}", model.cost(&TRAINING_SET));
        //println!();
    }
    println!("-------------------------------");
    println!("activation method: {ATC_M:?}");
    println!();
    println!("Final: {model:#?}");
    println!("cost: {}", model.cost(&TRAINING_SET));
    println!();
    for i in &TRAINING_SET {
        let input = vec![i[0], i[1]];
        let exp_output = i[2];
        let output = model.forward(&input);
        println!("{input:?}, exp_output: {exp_output}, output: {output}");
    }
    println!("-------------------------------");
    println!("a:");
    println!(" in_1 | in_2 ");
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "   {i}  |  {j}    = {}",
                ATC_M.apply_act(
                    model.neurons[0].bias
                        + f64::from(i) * model.neurons[0].weights[0]
                        + f64::from(j) * model.neurons[0].weights[1]
                )
            );
        }
    }
    println!("--------------");
    println!("b:");
    println!(" in_1 | in_2 ");
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "   {i}  |  {j}    = {}",
                ATC_M.apply_act(
                    model.neurons[1].bias
                        + f64::from(i) * model.neurons[1].weights[0]
                        + f64::from(j) * model.neurons[1].weights[1]
                )
            );
        }
    }
    println!("--------------");
    println!("c:");
    println!("   a  |  b   ");
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "   {i}  |  {j}    = {}",
                ATC_M.apply_act(
                    model.neurons[1].bias
                        + f64::from(i) * model.neurons[1].weights[0]
                        + f64::from(j) * model.neurons[1].weights[1]
                )
            );
        }
    }
}
