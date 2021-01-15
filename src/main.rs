extern crate nalgebra as na;

use num::abs;
use num_complex::Complex64;

enum Gate {
    I { t: usize },
    X { t: usize },
    Y { t: usize },
    Z { t: usize },
    H { t: usize },
    S { t: usize },
    T { t: usize },
    CX { c: usize, t: usize },
    // CY { c: usize, t: usize },
    // CZ { c: usize, t: usize },
    // SWAP { c: usize, t: usize },
    // CCX { c0: usize, c1: usize, t: usize },
}

fn controlled_gate(c: usize, t: usize, u: &na::DMatrix<Complex64>) -> na::DMatrix<Complex64> {
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
    fn matrix(&self) -> na::DMatrix<Complex64> {
        match *self {
            Gate::I { t: _ } => {
                na::DMatrix::identity_generic(na::Dynamic::new(2), na::Dynamic::new(2))
            }
            Gate::X { t: _ } => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(0., 0.),
                    Complex64::new(1., 0.),
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                ],
            ),
            Gate::Y { t: _ } => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(0., 0.),
                    Complex64::new(0., -1.),
                    Complex64::new(0., 1.),
                    Complex64::new(0., 0.),
                ],
            ),
            Gate::Z { t: _ } => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(-1., 0.),
                ],
            ),
            Gate::H { t: _ } => {
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
            Gate::S { t: _ } => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 1.),
                ],
            ),
            Gate::T { t: _ } => na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::from_polar(1., na::RealField::frac_pi_4()),
                ],
            ),
            Gate::CX { c, t } => controlled_gate(c, t, &Gate::X { t }.matrix()),
            // Gate::CY { c, t } => controlled_gate(c, t, &Gate::Y { t }.matrix()),
            // Gate::CZ { c, t } => controlled_gate(c, t, &Gate::Z { t }.matrix()),
            // Gate::SWAP { c, t } => {
            //     Gate::CX { c: c, t: t }.matrix()
            //         * Gate::CX { c: t, t: c }.matrix()
            //         * Gate::CX { c: c, t: t }.matrix()
            // }
        }
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
    let gate_cx = Gate::CX { c: 2, t: 0 };
    // let gate_cy = Gate::CY { c: 1, t: 2 };
    // let gate_cz = Gate::CZ { c: 1, t: 2 };
    // let gate_swap = Gate::SWAP { c: 1, t: 2 };

    // print!("Identity Gate\n{}", gate_i.matrix());
    // print!("Pauli-X Gate\n{}", gate_x.matrix());
    // print!("Pauli-Y Gate\n{}", gate_y.matrix());
    // print!("Pauli-Z Gate\n{}", gate_z.matrix());
    // print!("Hadamard Gate\n{}", gate_h.matrix());
    // print!("Phase Gate\n{}", gate_s.matrix());
    // print!("π/8 Gate\n{}", gate_t.matrix());
    print!("Controlled X Gate\n{}", gate_cx.matrix());
    // print!("Controlled Y Gate\n{}", gate_cy.matrix());
    // print!("Controlled Z Gate\n{}", gate_cz.matrix());
    // print!("Swap Gate\n{}", gate_swap.matrix());
}
