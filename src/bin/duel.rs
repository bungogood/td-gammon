use bkgm::Hypergammon;
use td_gammon::{
    duel::duel,
    evaluator::{HyperEvaluator, RandomEvaluator},
    fstate::FState,
};

pub mod train;

fn main() {
    let probs = duel::<FState<Hypergammon>>(
        HyperEvaluator::new().unwrap(),
        RandomEvaluator::new(),
        10000,
    );

    println!("{:?}", probs);
}
