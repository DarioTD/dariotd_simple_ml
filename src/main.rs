use simple_ml_gates_xor_sigm::{Model, ATC_M, TRAINING_SET};

fn main() {
    let mut m = Model::new_rand();
    let c = m.cost();
    println!("Original = m: {m:#?}");
    println!("cost: {c}");
    println!();
    let rate = 1e-2;
    let eps = 1e-3;

    for _ in 0..1_000_000 {
        let g = m.finite_diff(eps);
        m.apply_diff(&g, rate);
        //println!("{i} = m: {m:#?}");
        //println!("cost: {}", m.cost());
        //println!();
    }
    println!("-------------------------------");
    println!("activation method: {ATC_M:?}");
    println!();
    println!("Final = m: {m:#?}");
    println!("cost: {}", m.cost());
    println!();
    for i in TRAINING_SET {
        let x1 = i[0];
        let x2 = i[1];
        let ex_out = i[2];
        let y = m.forward(x1, x2);
        println!("x1: {x1}, x2: {x2}, ex_out: {ex_out}, y: {y}");
    }
    println!("-------------------------------");
    println!("a:");
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{i} | {j} = {}",
                ATC_M
                    .activate(f64::from(i).mul_add(m.n[0].w1, f64::from(j) * m.n[0].w2) + m.n[0].b)
            );
        }
    }
    println!("b:");
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{i} | {j} = {}",
                ATC_M
                    .activate(f64::from(i).mul_add(m.n[1].w1, f64::from(j) * m.n[1].w2) + m.n[1].b)
            );
        }
    }
    println!("c:");
    println!("a | b");
    println!("--------------");
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{i} | {j} = {}",
                ATC_M
                    .activate(f64::from(i).mul_add(m.n[2].w1, f64::from(j) * m.n[2].w2) + m.n[2].b)
            );
        }
    }
}
