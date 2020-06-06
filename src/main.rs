extern crate nalgebra as na;

use num_complex::Complex64;

enum Gate {
    I { t: u64 },
    X { t: u64 },
    Y { t: u64 },
    Z { t: u64 },
    H { t: u64 },
    S { t: u64 },
    T { t: u64 },
    CX { c: u64, t: u64 },
    CY { c: u64, t: u64 },
    CZ { c: u64, t: u64 },
    SWAP { c: u64, t: u64 },
    CCX { c0: u64, c1: u64, t: u64 },
}

fn controlled_gate(c: u64, t: u64, u: &na::DMatrix<Complex64>) -> na::DMatrix<Complex64> {
    let mut c0part = na::DMatrix::identity_generic(na::Dynamic::new(1), na::Dynamic::new(1));
    let mut c1part = na::DMatrix::identity_generic(na::Dynamic::new(1), na::Dynamic::new(1));
    for q in na::min(c, t)..=na::max(c, t) {
        if q == c {
            c0part = c0part.kronecker(&na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(1., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                ],
            ));
            c1part = c1part.kronecker(&na::DMatrix::from_vec_generic(
                na::Dynamic::new(2),
                na::Dynamic::new(2),
                vec![
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(0., 0.),
                    Complex64::new(1., 0.),
                ],
            ));
        } else if q == t {
            c0part = c0part.kronecker(&Gate::I { t }.matrix());
            c1part = c1part.kronecker(u);
        } else {
            c0part = c0part.kronecker(&Gate::I { t }.matrix());
            c1part = c1part.kronecker(&Gate::I { t }.matrix());
        }
    }
    c0part + c1part
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
                    Complex64::from_polar(&1., &na::RealField::frac_pi_4()),
                ],
            ),
            Gate::CX { c, t } => controlled_gate(c, t, &Gate::X { t }.matrix()),
            Gate::CY { c, t } => controlled_gate(c, t, &Gate::Y { t }.matrix()),
            Gate::CZ { c, t } => controlled_gate(c, t, &Gate::Z { t }.matrix()),
            Gate::SWAP { c, t } => {
                Gate::CX { c: c, t: t }.matrix()
                    * Gate::CX { c: t, t: c }.matrix()
                    * Gate::CX { c: c, t: t }.matrix()
            }
            Gate::CCX { c0, c1, t } => {
                // TODO: Currently won't work for more than 3 qubits, needs rework
                Gate::CX { c: c1, t: c0 }
                    .matrix()
                    .kronecker(&Gate::I { t: t }.matrix())
                    * Gate::T { t: c0 }
                        .matrix()
                        .kronecker(&Gate::T { t: c1 }.matrix().adjoint())
                        .kronecker(&Gate::I { t: t }.matrix())
                    * Gate::CX { c: c1, t: c0 }
                        .matrix()
                        .kronecker(&Gate::H { t: t }.matrix())
                    * Gate::I { t: c0 }
                        .matrix()
                        .kronecker(&Gate::T { t: c1 }.matrix())
                        .kronecker(&Gate::T { t: t }.matrix())
                    * Gate::CX { c: t, t: c0 }.matrix()
                    * Gate::I { t: c0 }
                        .matrix()
                        .kronecker(&Gate::I { t: c1 }.matrix())
                        .kronecker(&Gate::T { t: t }.matrix().adjoint())
                    * Gate::I { t: c0 }
                        .matrix()
                        .kronecker(&Gate::CX { c: t, t: c1 }.matrix())
                    * Gate::I { t: c0 }
                        .matrix()
                        .kronecker(&Gate::I { t: c1 }.matrix())
                        .kronecker(&Gate::T { t: t }.matrix())
                    * Gate::CX { c: t, t: c0 }.matrix()
                    * Gate::I { t: c0 }
                        .matrix()
                        .kronecker(&Gate::I { t: c1 }.matrix())
                        .kronecker(&Gate::T { t: t }.matrix().adjoint())
                    * Gate::I { t: c0 }
                        .matrix()
                        .kronecker(&Gate::CX { c: t, t: c1 }.matrix())
                    * Gate::I { t: c0 }
                        .matrix()
                        .kronecker(&Gate::I { t: c1 }.matrix())
                        .kronecker(&Gate::H { t: t }.matrix())
            }
        }
    }
}

fn main() {
    let gate_i = Gate::I { t: 1 };
    let gate_x = Gate::X { t: 1 };
    let gate_y = Gate::Y { t: 1 };
    let gate_z = Gate::Z { t: 1 };
    let gate_h = Gate::H { t: 1 };
    let gate_s = Gate::S { t: 1 };
    let gate_t = Gate::T { t: 1 };
    let gate_cx = Gate::CX { c: 1, t: 2 };
    let gate_cy = Gate::CY { c: 1, t: 2 };
    let gate_cz = Gate::CZ { c: 1, t: 2 };
    let gate_swap = Gate::SWAP { c: 1, t: 2 };
    let gate_ccx = Gate::CCX { c0: 3, c1: 2, t: 1 };

    print!("Identity Gate\n{}", gate_i.matrix());
    print!("Pauli-X Gate\n{}", gate_x.matrix());
    print!("Pauli-Y Gate\n{}", gate_y.matrix());
    print!("Pauli-Z Gate\n{}", gate_z.matrix());
    print!("Hadamard Gate\n{}", gate_h.matrix());
    print!("Phase Gate\n{}", gate_s.matrix());
    print!("π/8 Gate\n{}", gate_t.matrix());
    print!("Controlled X Gate\n{}", gate_cx.matrix());
    print!("Controlled Y Gate\n{}", gate_cy.matrix());
    print!("Controlled Z Gate\n{}", gate_cz.matrix());
    print!("Swap Gate\n{}", gate_swap.matrix());
    print!("Toffoli Gate\n{}", gate_ccx.matrix());
}
