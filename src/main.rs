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
}

impl Gate {
    fn matrix(&self) -> na::DMatrix<Complex64> {
        match self {
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
            Gate::Y { t: _ } => unimplemented!(),
            Gate::Z { t: _ } => unimplemented!(),
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
            Gate::S { t: _ } => unimplemented!(),
            Gate::T { t: _ } => unimplemented!(),
            Gate::CX { c: _, t: _ } => unimplemented!(),
            Gate::CY { c: _, t: _ } => unimplemented!(),
            Gate::CZ { c: _, t: _ } => unimplemented!(),
            Gate::SWAP { c: _, t: _ } => unimplemented!(),
        }
    }
}

fn main() {
    let gate_i = Gate::I { t: 1 };
    let gate_h = Gate::H { t: 1 };

    print!("{}", gate_i.matrix().kronecker(&gate_h.matrix()));
}
