use bkgm::{Dice, State};
use fastrand;

pub trait Evaluator<G: State>: Sized {
    fn best_position(&self, pos: &G, dice: &Dice) -> G;
}

pub struct RandomEvaluator;

impl RandomEvaluator {
    pub fn new() -> Self {
        Self
    }
}

impl<G: State> Evaluator<G> for RandomEvaluator {
    fn best_position(&self, pos: &G, dice: &Dice) -> G {
        let possible_positions = pos.possible_positions(dice);
        let index = fastrand::usize(..possible_positions.len());
        possible_positions[index]
    }
}
