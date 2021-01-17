extern crate nalgebra as na;

use rand::distributions::WeightedIndex;
use rand::prelude::*;

use num::abs;
use num_complex::Complex64;

use std::collections::HashSet;

pub enum Gate {
    I(usize),
    X(usize),
    Y(usize),
    Z(usize),
    H(usize),
    S(usize),
    T(usize),
    CX(usize, usize),
    SWAP(usize, usize),
}

impl Gate {
    fn controlled_x_gate(c: usize, t: usize) -> na::DMatrix<Complex64> {
        let control: usize = c - c.min(t);
        let target: usize = t - t.min(c);
        let num_qubits: usize = (abs(control as i64 - target as i64) + 1) as usize;
        let num_states: usize = 1 << num_qubits;
        let control_mask: usize = num_states >> (control + 1);
        let target_mask: usize = num_states >> (target + 1);
        let mut matrix = na::DMatrix::<Complex64>::zeros_generic(
            na::Dynamic::new(num_states),
            na::Dynamic::new(num_states),
        );
        for i in 0..num_states {
            let output = na::DMatrix::<Complex64>::from_fn_generic(
                na::Dynamic::new(num_states),
                na::Dynamic::new(1),
                |row, _| {
                    if i & control_mask == 0 {
                        if row != i {
                            Complex64::new(0., 0.)
                        } else {
                            Complex64::new(1., 0.)
                        }
                    } else {
                        if row != i ^ target_mask {
                            Complex64::new(0., 0.)
                        } else {
                            Complex64::new(1., 0.)
                        }
                    }
                },
            );
            let input = na::DMatrix::<Complex64>::from_fn_generic(
                na::Dynamic::new(1),
                na::Dynamic::new(num_states),
                |_, col| {
                    if col != i {
                        Complex64::new(0., 0.)
                    } else {
                        Complex64::new(1., 0.)
                    }
                },
            );
            matrix += output.kronecker(&input);
        }
        matrix
    }

    pub fn matrix(&self) -> na::DMatrix<Complex64> {
        match *self {
            Gate::I(_) => na::DMatrix::identity_generic(na::Dynamic::new(2), na::Dynamic::new(2)),
            Gate::X(_) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(0., 0.),
                    Complex64::new(1., 0.),
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                ],
            ),
            Gate::Y(_) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(0., 0.),
                    Complex64::new(0., -1.),
                    Complex64::new(0., 1.),
                    Complex64::new(0., 0.),
                ],
            ),
            Gate::Z(_) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(-1., 0.),
                ],
            ),
            Gate::H(_) => {
                na::DMatrix::from_vec_generic(
                    na::Dynamic::new(2),
                    na::Dynamic::new(2),
                    vec![
                        Complex64::new(1., 0.),
                        Complex64::new(1., 0.),
                        Complex64::new(1., 0.),
                        Complex64::new(-1., 0.),
                    ],
                ) / Complex64::new(2., 0.).sqrt()
            }
            Gate::S(_) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 1.),
                ],
            ),
            Gate::T(_) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::from_polar(1., na::RealField::frac_pi_4()),
                ],
            ),
            Gate::CX(c, t) => Self::controlled_x_gate(c, t),
            Gate::SWAP(c, t) => {
                Gate::CX(c, t).matrix() * Gate::CX(t, c).matrix() * Gate::CX(c, t).matrix()
            }
        }
    }

    pub fn num_qubits(&self) -> usize {
        match *self {
            Gate::I(_) => 1,
            Gate::X(_) => 1,
            Gate::Y(_) => 1,
            Gate::Z(_) => 1,
            Gate::H(_) => 1,
            Gate::S(_) => 1,
            Gate::T(_) => 1,
            Gate::CX(c, t) => (abs(c as i64 - t as i64) + 1) as usize,
            Gate::SWAP(c, t) => (abs(c as i64 - t as i64) + 1) as usize,
        }
    }

    pub fn position(&self) -> usize {
        match *self {
            Gate::I(t) => t,
            Gate::X(t) => t,
            Gate::Y(t) => t,
            Gate::Z(t) => t,
            Gate::H(t) => t,
            Gate::S(t) => t,
            Gate::T(t) => t,
            Gate::CX(c, t) => c.min(t),
            Gate::SWAP(c, t) => c.min(t),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    matrix: na::DMatrix<Complex64>,
    num_qubits: usize,
    num_states: usize,
}

impl QuantumCircuit {
    pub fn new(num_qubits: usize, initial_state: usize) -> Self {
        Self {
            matrix: na::DMatrix::<Complex64>::from_fn_generic(
                na::Dynamic::new(1 << num_qubits),
                na::Dynamic::new(1),
                |row, _| {
                    if row != initial_state {
                        Complex64::new(0., 0.)
                    } else {
                        Complex64::new(1., 0.)
                    }
                },
            )
            .normalize(),
            num_qubits: num_qubits,
            num_states: 1 << num_qubits,
        }
    }

    pub fn from_states(initial_states: Vec<usize>) -> Self {
        let num_qubits: usize;

        // Get qubits needed for max state
        if let Some(max) = initial_states.iter().max() {
            num_qubits = (*max as f64).log2() as usize + 1;
        } else {
            panic!("Vector is empty!");
        }

        // Check if vector has no repeated elements
        assert!({
            let mut uniq = HashSet::new();
            initial_states.iter().all(move |x| uniq.insert(x))
        });

        let num_states = 1 << num_qubits;

        Self {
            matrix: na::DMatrix::<Complex64>::from_fn_generic(
                na::Dynamic::new(num_states),
                na::Dynamic::new(1),
                |row, _| {
                    if initial_states.iter().any(|&i| i == row) {
                        Complex64::new(1., 0.)
                    } else {
                        Complex64::new(0., 0.)
                    }
                },
            )
            .normalize(),
            num_qubits: num_qubits,
            num_states: num_states,
        }
    }

    pub fn from_state_vector(state_vector: Vec<Complex64>) -> Self {
        let len = state_vector.len();

        // Check if not empty
        assert!(len != 0);

        // Check if power of 2
        assert!((len != 0) && ((len & (len - 1)) == 0));

        let num_qubits = (len as f64).log2() as usize;
        let num_states = 1 << num_qubits;

        Self {
            matrix: na::DMatrix::from_vec_generic(
                na::Dynamic::new(num_states),
                na::Dynamic::new(1),
                state_vector,
            )
            .normalize(),
            num_qubits: num_qubits,
            num_states: num_states,
        }
    }

    pub fn state(&self) -> na::DMatrix<Complex64> {
        self.matrix.clone()
    }

    pub fn apply_gate(&mut self, gate: Gate) {
        let identity = Gate::I(0);

        let mut matrix = na::DMatrix::identity_generic(na::Dynamic::new(1), na::Dynamic::new(1));

        for _ in 0..gate.position() {
            matrix = matrix.kronecker(&identity.matrix());
        }

        matrix = matrix.kronecker(&gate.matrix());

        for _ in (gate.position() + gate.num_qubits())..self.num_qubits {
            matrix = matrix.kronecker(&identity.matrix());
        }

        self.matrix = matrix * self.matrix.clone();

        self.normalize();
    }

    pub fn measure(&mut self) -> usize {
        let mut rng = thread_rng();

        let mut items = Vec::<(usize, f64)>::new();

        for (i, num) in self.matrix.iter().enumerate() {
            items.push((i, num.norm_sqr()))
        }
        let dist = WeightedIndex::new(items.iter().map(|item| item.1)).unwrap();
        let new_state = items[dist.sample(&mut rng)].0;

        self.matrix = na::DMatrix::<Complex64>::from_fn_generic(
            na::Dynamic::new(self.num_states),
            na::Dynamic::new(1),
            |row, _| {
                if row != new_state {
                    Complex64::new(0., 0.)
                } else {
                    Complex64::new(1., 0.)
                }
            },
        );

        self.normalize();

        new_state
    }

    fn normalize(&mut self) {
        self.matrix = self.matrix.normalize();
    }
}
