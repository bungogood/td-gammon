use std::marker::PhantomData;

use crate::dicegen::{DiceGen, FastrandDice};
use crate::evaluator::Evaluator;
use crate::probabilities::{Probabilities, ResultCounter};
use bkgm::GameState::{GameOver, Ongoing};
use bkgm::State;
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Clone, Copy)]
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
    let bar = ProgressBar::new(rounds as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{pos:>5}/{len:5} {msg} {wide_bar} {eta}")
            .unwrap(),
    );
    for _ in 1..=rounds {
        let outcome = duel.single_duel(&mut FastrandDice::new());
        results = results.combine(&outcome);
        let probs = results.probabilities();

        let message = format!(
            "wn:{:.2}% wg:{:.2}% wb:{:.2}% lg:{:.2}% lb:{:.2}%",
            probs.win_prob() * 100.0,
            probs.win_g * 100.0,
            probs.win_b * 100.0,
            probs.lose_g * 100.0,
            probs.lose_b * 100.0,
        );
        bar.set_message(message);
        bar.inc(1);
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
        let mut dice = dice_gen.first_roll();
        while !(pos1_finished && pos2_finished) {
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
            dice = dice_gen.roll();
            iteration += 1;
        }
        debug_assert!(counter.sum() == 2, "Each duel should have two game results");
        counter
    }
}
