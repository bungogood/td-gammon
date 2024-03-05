use std::io::{stdout, Write};

use crate::{
    dicegen::DiceGen,
    duel::duel,
    evaluator::{Evaluator, HyperEvaluator, PubEval, RandomEvaluator},
    fstate::FState,
};
use bkgm::{
    GameState::{GameOver, Ongoing},
    Hypergammon, State,
};
use burn::{module::Module, record::NoStdTrainingRecorder};
use burn::{
    optim::{momentum::MomentumConfig, GradientsParams, Optimizer, SgdConfig},
    tensor::{backend::AutodiffBackend, Data, Tensor},
};

use crate::{dicegen::FastrandDice, model::TDModel};

pub struct TDConfig {
    learning_rate: f64,
    td_decay: f64,
}

impl TDConfig {
    pub fn new(learning_rate: f64, td_decay: f64) -> Self {
        Self {
            learning_rate,
            td_decay,
        }
    }
}

pub struct TDTrainer<B: AutodiffBackend> {
    device: B::Device,
    optim: SgdConfig,
    config: TDConfig,
}

impl<B: AutodiffBackend> TDTrainer<B> {
    pub fn new(device: B::Device, config: TDConfig) -> Self {
        let optim = SgdConfig::new().with_momentum(Some(
            MomentumConfig::new()
                .with_dampening(0.0)
                .with_momentum(config.td_decay),
        ));
        Self {
            device,
            optim,
            config,
        }
    }

    fn get_value<G: State + Send>(&self, state: &FState<G>, model: &TDModel<B>) -> Tensor<B, 1> {
        let state = if state.turn { *state } else { state.flip() };
        match state.game_state() {
            GameOver(result) => model.from_result(result, &self.device),
            Ongoing => model.forward_pos(state, &self.device),
        }
    }

    fn train_game<G: State + Send>(&mut self, model: TDModel<B>) -> TDModel<B> {
        let mut optim = self.optim.init();
        let mut model = model;

        let mut dicegen = FastrandDice::new();
        let mut dice = Some(dicegen.first_roll());
        let mut state = FState::<G>::new();

        while state.game_state() == Ongoing {
            let cur_value = self.get_value(&state, &model);
            let grads = GradientsParams::from_grads(cur_value.backward(), &model);
            if dice == None {
                dice = Some(dicegen.roll());
            }
            state = model.best_position(&state, &dice.unwrap());
            dice = None;
            let next_value = self.get_value(&state, &model);
            let td_error = next_value - cur_value.clone();
            let data: Data<f64, 1> = td_error.to_data().convert();
            model = optim.step(-self.config.learning_rate * data.value[0], model, grads);
        }

        model
    }

    pub fn train<G: State + Send>(&mut self, model: TDModel<B>, num_episodes: usize) -> TDModel<B> {
        // self.train_one(model.clone());
        let mut model = model;
        let mut best_model = model.clone();
        let mut best_ep = 0;
        let mut ep = 0;
        while ep < num_episodes {
            model = self.train_game::<G>(model.clone());
            if ep % 100 == 0 {
                print!("\rEpisode: {}", ep);
                stdout().flush().unwrap();
            }

            ep += 1;

            if ep % 5_000 == 0 {
                let probs = duel::<FState<G>>(model.clone(), RandomEvaluator::new(), 1000);
                println!(
                    "Random Equity: {:.3} ({:.1}%). {:?}",
                    probs.equity(),
                    probs.win_prob() * 100.0,
                    probs,
                );
                let probs = duel::<FState<G>>(model.clone(), best_model.clone(), 1000);
                println!(
                    "Prev Equity: {:.3} ({:.1}%). {:?}",
                    probs.equity(),
                    probs.win_prob() * 100.0,
                    probs,
                );
                if probs.win_prob() > 0.53 {
                    best_model = model.clone();
                    best_ep = ep;
                }
            }

            if ep - best_ep > 100_000 {
                println!("No improvement. Reseting.");
                model = best_model.clone();
                ep = best_ep;
            }

            if ep % 25_000 == 0 {
                let probs = duel::<FState<Hypergammon>>(
                    model.clone(),
                    HyperEvaluator::new().unwrap(),
                    1000,
                );
                println!(
                    "Hyper Equity: {:.3} ({:.1}%). {:?}",
                    probs.equity(),
                    probs.win_prob() * 100.0,
                    probs,
                );
                let probs = duel::<FState<Hypergammon>>(model.clone(), PubEval::new(), 1000);
                println!(
                    "Pub Eval Equity: {:.3} ({:.1}%). {:?}",
                    probs.equity(),
                    probs.win_prob() * 100.0,
                    probs,
                );
                // let probs = duel::<FState<G>>(model.clone(), prev_model.clone(), 1000);
                // println!(
                //     "Prev Equity: {:.3} ({:.1}%). {:?}",
                //     probs.equity(),
                //     probs.win_prob() * 100.0,
                //     probs,
                // );
                // if probs.win_prob() < 0.5 {
                //     model = prev_model.clone();
                //     println!("Not improving");
                // } else {
                //     println!("Saving model");
                //     model
                //         .clone()
                //         .save_file(
                //             format!("model/td-next/games-{}", ep),
                //             &NoStdTrainingRecorder::new(),
                //         )
                //         .expect("Failed to save model");
                // }
                // prev_model = model.clone();
            }
            // if ep % 2_000 == 0 {
            //     println!("Saving model");
            //     model
            //         .clone()
            //         .save_file(
            //             format!("model/td/games-{}", ep),
            //             &NoStdTrainingRecorder::new(),
            //         )
            //         .expect("Failed to save model");
            //     let probs = duel::duel::<FState<Hypergammon>>(model.clone(), prev_model, 1000);
            //     println!(
            //         "Equity: {:.3} ({:.1}%). {:?}",
            //         probs.equity(),
            //         probs.win_prob() * 100.0,
            //         probs,
            //     );
            //     prev_model = model.clone();
            // }
        }
        model
    }
}