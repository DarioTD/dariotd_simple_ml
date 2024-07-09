#[allow(dead_code)]
pub enum TrainingDataSet {
    Not,
    And,
    Nand,
    Or,
    Nor,
    Xor,
    Nxor,
    BinToState,
    StateToBin,
}

pub fn training_data_set(ts: &TrainingDataSet) -> Vec<(Vec<f64>, Vec<f64>)> {
    let not = vec![(vec![0.0], vec![1.0]), (vec![1.0], vec![0.0])];
    let and = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];
    let nand = vec![
        (vec![0.0, 0.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    let or = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];
    let nor = vec![
        (vec![0.0, 0.0], vec![1.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    let xor = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    let nxor = vec![
        (vec![0.0, 0.0], vec![1.0]),
        (vec![1.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![0.0]),
        (vec![1.0, 1.0], vec![1.0]),
    ];
    let bin_to_state = vec![
        (
            vec![0.0, 0.0, 0.0, 0.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![0.0, 0.0, 0.0, 1.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ),
        (
            vec![0.0, 0.0, 1.0, 0.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
        ),
        (
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
        ),
        (
            vec![0.0, 1.0, 0.0, 0.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![0.0, 1.0, 0.0, 1.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![0.0, 1.0, 1.0, 0.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![0.0, 1.0, 1.0, 1.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![1.0, 0.0, 0.0, 0.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![1.0, 0.0, 0.0, 1.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![1.0, 0.0, 1.0, 0.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![1.0, 0.0, 1.0, 1.0],
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![1.0, 1.0, 0.0, 0.0],
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![1.0, 1.0, 0.0, 1.0],
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![1.0, 1.0, 1.0, 0.0],
            vec![
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
        (
            vec![1.0, 1.0, 1.0, 1.0],
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ),
    ];
    let state_to_bin = vec![
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![0.0, 0.0, 0.0, 0.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
            vec![0.0, 0.0, 0.0, 1.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ],
            vec![0.0, 0.0, 1.0, 0.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            ],
            vec![0.0, 0.0, 1.0, 1.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            ],
            vec![0.0, 1.0, 0.0, 0.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![0.0, 1.0, 0.0, 1.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![0.0, 1.0, 1.0, 0.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![0.0, 1.0, 1.0, 1.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![1.0, 0.0, 0.0, 0.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![1.0, 0.0, 1.0, 0.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![1.0, 0.0, 1.0, 1.0],
        ),
        (
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![1.0, 1.0, 0.0, 0.0],
        ),
        (
            vec![
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![1.0, 1.0, 0.0, 1.0],
        ),
        (
            vec![
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![1.0, 1.0, 1.0, 0.0],
        ),
        (
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            vec![1.0, 1.0, 1.0, 1.0],
        ),
    ];
    match ts {
        TrainingDataSet::Not => not,
        TrainingDataSet::And => and,
        TrainingDataSet::Nand => nand,
        TrainingDataSet::Or => or,
        TrainingDataSet::Nor => nor,
        TrainingDataSet::Xor => xor,
        TrainingDataSet::Nxor => nxor,
        TrainingDataSet::BinToState => bin_to_state,
        TrainingDataSet::StateToBin => state_to_bin,
    }
}
