use bkgm::Hypergammon;
use td_gammon::{
    duel::duel,
    evaluator::{HyperEvaluator, PubEval, RandomEvaluator},
    fstate::FState,
};

pub mod train;

fn main() {
    let probs = duel::<FState<Hypergammon>>(
        HyperEvaluator::new().unwrap(),
        // HyperEvaluator::new().unwrap(),
        // PubEval::new(),
        // PubEval::new(),
        RandomEvaluator::new(),
        // RandomEvaluator::new(),
        1_000_000,
    );

    println!(
        "Equity: {:.3} ({:.2}%) {:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
        probs.equity(),
        probs.win_prob() * 100.0,
        probs.win_n * 100.0,
        probs.win_g * 100.0,
        probs.win_b * 100.0,
        probs.lose_n * 100.0,
        probs.lose_g * 100.0,
        probs.lose_b * 100.0
    );
}
