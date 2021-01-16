extern crate nalgebra as na;

use num::abs;
use num_complex::Complex64;

enum Gate {
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

impl Gate {
    pub fn matrix(&self) -> na::DMatrix<Complex64> {
        match *self {
            Gate::I(t) => na::DMatrix::identity_generic(na::Dynamic::new(2), na::Dynamic::new(2)),
            Gate::X(t) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(0., 0.),
                    Complex64::new(1., 0.),
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                ],
            ),
            Gate::Y(t) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(0., 0.),
                    Complex64::new(0., -1.),
                    Complex64::new(0., 1.),
                    Complex64::new(0., 0.),
                ],
            ),
            Gate::Z(t) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(-1., 0.),
                ],
            ),
            Gate::H(t) => {
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
            Gate::S(t) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 1.),
                ],
            ),
            Gate::T(t) => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::from_polar(1., na::RealField::frac_pi_4()),
                ],
            ),
            Gate::CX(c, t) => controlled_x_gate(c, t),
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

#[derive(Debug)]
struct QuantumCircuit {
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
            ),
            num_qubits: num_qubits,
            num_states: 1 << num_qubits,
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
    }
}

fn main() {
    // let gate_i = Gate::I { t: 1 };
    // let gate_x = Gate::X { t: 1 };
    // let gate_y = Gate::Y { t: 1 };
    // let gate_z = Gate::Z { t: 1 };
    // let gate_h = Gate::H { t: 1 };
    // let gate_s = Gate::S { t: 1 };
    // let gate_t = Gate::T { t: 1 };
    // let gate_cx = Gate::CX { c: 0, t: 1 };
    // let gate_swap = Gate::SWAP { c: 0, t: 1 };

    // print!("Identity Gate\n{}", gate_i.matrix());
    // print!("Pauli-X Gate\n{}", gate_x.matrix());
    // print!("Pauli-Y Gate\n{}", gate_y.matrix());
    // print!("Pauli-Z Gate\n{}", gate_z.matrix());
    // print!("Hadamard Gate\n{}", gate_h.matrix());
    // print!("Phase Gate\n{}", gate_s.matrix());
    // print!("π/8 Gate\n{}", gate_t.matrix());
    // print!("Controlled X Gate\n{}", gate_cx.matrix());
    // print!("Swap Gate\n{}", gate_swap.matrix());

    let mut circuit = QuantumCircuit::new(3, 0);
    print!("Estado Inicial\n{}", circuit.state());

    circuit.apply_gate(Gate::H(0));
    print!("Estado após Hadamard\n{}", circuit.state());

    circuit.apply_gate(Gate::CX(0, 2));
    print!("Estado após CNOT\n{}", circuit.state());
}
