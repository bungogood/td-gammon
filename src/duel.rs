use std::marker::PhantomData;

use crate::dicegen::{DiceGen, FastrandDice};
use crate::evaluator::Evaluator;
use crate::probabilities::{Probabilities, ResultCounter};
use bkgm::GameState::{GameOver, Ongoing};
use bkgm::State;
use indicatif::ProgressIterator;

pub struct Duel<T: Evaluator<G>, U: Evaluator<G>, G: State> {
    evaluator1: T,
    evaluator2: U,
    phantom: PhantomData<G>,
}

pub fn duel<G: State>(
    evaluator1: impl Evaluator<G>,
    evaluator2: impl Evaluator<G>,
    rounds: usize,
) -> Probabilities {
    let duel = Duel::new(evaluator1, evaluator2);
    let mut results = ResultCounter::default();
    for _ in (0..rounds).progress() {
        let outcome = duel.single_duel(&mut FastrandDice::new());
        results = results.combine(&outcome);
    }
    results.probabilities()
}

/// Let two `PartialEvaluator`s duel each other. A bit quick and dirty.
impl<T: Evaluator<G>, U: Evaluator<G>, G: State> Duel<T, U, G> {
    #[allow(clippy::new_without_default)]
    pub fn new(evaluator1: T, evaluator2: U) -> Self {
        Duel {
            evaluator1,
            evaluator2,
            phantom: PhantomData,
        }
    }

    // pub fn

    /// The two `PartialEvaluator`s will play twice each against each other.
    /// Either `PartialEvaluator` will start once and play with the same dice as vice versa.
    pub fn single_duel<V: DiceGen>(&self, dice_gen: &mut V) -> ResultCounter {
        let mut pos1 = G::new();
        let mut pos2 = G::new();
        let mut iteration = 1;
        let mut pos1_finished = false;
        let mut pos2_finished = false;
        let mut counter = ResultCounter::default();
        while !(pos1_finished && pos2_finished) {
            let dice = dice_gen.roll();
            match pos1.game_state() {
                Ongoing => {
                    pos1 = if iteration % 2 == 0 {
                        self.evaluator1.best_position(&pos1, &dice)
                    } else {
                        self.evaluator2.best_position(&pos1, &dice)
                    };
                }
                GameOver(result) => {
                    if !pos1_finished {
                        pos1_finished = true;
                        let result = if iteration % 2 == 0 {
                            result
                        } else {
                            result.reverse()
                        };
                        counter.add(result);
                    }
                }
            }
            match pos2.game_state() {
                Ongoing => {
                    pos2 = if iteration % 2 == 0 {
                        self.evaluator2.best_position(&pos2, &dice)
                    } else {
                        self.evaluator1.best_position(&pos2, &dice)
                    };
                }
                GameOver(result) => {
                    if !pos2_finished {
                        pos2_finished = true;
                        let result = if iteration % 2 == 0 {
                            result.reverse()
                        } else {
                            result
                        };
                        counter.add(result);
                    }
                }
            }
            iteration += 1;
        }
        debug_assert!(counter.sum() == 2, "Each duel should have two game results");
        counter
    }
}
