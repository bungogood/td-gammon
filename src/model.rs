use std::path::PathBuf;

use crate::{evaluator::Evaluator, fstate::FState, inputs::Inputs};
use bkgm::GameState::{GameOver, Ongoing};
use bkgm::{dice::ALL_21, position, Dice, GameResult, Position, State};
use burn::{
    data,
    module::Module,
    nn::{
        self,
        loss::{MSELoss, Reduction::Mean},
    },
    record::{NoStdTrainingRecorder, Recorder},
    tensor::{
        self,
        activation::sigmoid,
        backend::{AutodiffBackend, Backend},
        Data, Tensor,
    },
    train::RegressionOutput,
};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

#[derive(Module, Debug)]
pub struct TDModel<B: Backend> {
    fc1: nn::Linear<B>,
    output: nn::Linear<B>,
}

impl<B: Backend> Default for TDModel<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(&device)
    }
}

impl<B: Backend> TDModel<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            fc1: nn::LinearConfig::new(202, 160).init(device),
            output: nn::LinearConfig::new(160, 1).init(device),
        }
    }

    pub fn new_from(record: TDModelRecord<B>) -> Self {
        Self {
            fc1: nn::LinearConfig::new(202, 160).init_with(record.fc1),
            output: nn::LinearConfig::new(160, 1).init_with(record.output),
        }
    }

    pub fn init_with(device: B::Device, model_path: &PathBuf) -> Self {
        let record = NoStdTrainingRecorder::new()
            .load(model_path.into(), &device)
            .expect("Failed to load model");
        Self::new_from(record)
    }

    fn inputs(&self, position: &bkgm::Position) -> Data<f32, 1> {
        Data::<f32, 1>::from(Inputs::from_position(position).to_vec().as_slice())
    }

    pub fn input_tensor(&self, device: &B::Device, positions: Vec<Position>) -> Tensor<B, 2> {
        let tensor_pos = positions
            .iter()
            .map(|item| self.inputs(&item))
            .map(|data| Tensor::<B, 1>::from_data(data.convert(), device))
            .map(|tensor| tensor.reshape([1, 202]))
            .collect();

        Tensor::cat(tensor_pos, 0)
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(input);
        let x = sigmoid(x);
        let x = self.output.forward(x);
        sigmoid(x)
    }

    pub fn forward_pos<G: bkgm::State>(&self, position: G, device: &B::Device) -> Tensor<B, 1> {
        let inputs = self.input_tensor(device, vec![position.position()]);
        let output = self.forward(inputs);
        output.reshape([1])
    }

    pub fn result_value(&self, result: GameResult) -> f32 {
        match result {
            GameResult::WinNormal => 1.0,
            GameResult::WinGammon => 1.0,
            GameResult::WinBackgammon => 1.0,
            GameResult::LoseNormal => 0.0,
            GameResult::LoseGammon => 0.0,
            GameResult::LoseBackgammon => 0.0,
        }
    }

    pub fn from_result(&self, result: GameResult, device: &B::Device) -> Tensor<B, 1> {
        let data = Data::<f32, 1>::from([self.result_value(result)].as_slice());
        Tensor::<B, 1>::from_data(data.convert(), device)
    }

    fn finder<G: State + Send>(
        &self,
        maxer: bool,
        pos: &FState<G>,
        dice: &Dice,
    ) -> (FState<G>, f32) {
        let device = B::Device::default();

        let positions = pos.possible_positions(dice);

        if pos.turn {
            let inputs =
                self.input_tensor(&device, positions.iter().map(|p| p.position()).collect());
            let outputs = self.forward(inputs);

            let data: Data<f32, 2> = outputs.into_data().convert();

            if maxer {
                positions
                    .iter()
                    .enumerate()
                    .map(|(i, pos)| (*pos, data.value[i]))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
            } else {
                positions
                    .iter()
                    .enumerate()
                    .map(|(i, pos)| (*pos, data.value[i]))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
            }
        } else {
            let inputs = self.input_tensor(
                &device,
                positions.iter().map(|p| p.position().flip()).collect(),
            );
            let outputs = self.forward(inputs);

            let data: Data<f32, 2> = outputs.into_data().convert();

            if maxer {
                positions
                    .iter()
                    .enumerate()
                    .map(|(i, pos)| (*pos, data.value[i]))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
            } else {
                positions
                    .iter()
                    .enumerate()
                    .map(|(i, pos)| (*pos, data.value[i]))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
            }
        }
    }

    fn nply<G: State + Send>(
        &self,
        depth: u8,
        maxer: bool,
        pos: &FState<G>,
        dice: &Dice,
    ) -> (FState<G>, f32) {
        if depth == 1 {
            return self.finder(maxer, pos, dice);
        }

        if pos.turn {
            if maxer {
                pos.possible_positions(dice)
                    .iter()
                    .map(|p| match p.game_state() {
                        GameOver(result) => (
                            *p,
                            if !pos.turn {
                                self.result_value(result)
                            } else {
                                1.0 - self.result_value(result)
                            },
                        ), // check p.turn
                        Ongoing => {
                            let mut nprobs = 0.0;
                            let mut total = 0.0;
                            for (dice, n) in ALL_21 {
                                let (_, v) = self.nply(depth - 1, !maxer, p, &dice);
                                total += v;
                                nprobs += n;
                            }
                            (*p, total / nprobs)
                        }
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
            } else {
                pos.possible_positions(dice)
                    .iter()
                    .map(|p| match p.game_state() {
                        GameOver(result) => (
                            *p,
                            if !pos.turn {
                                self.result_value(result)
                            } else {
                                1.0 - self.result_value(result)
                            },
                        ), // check p.turn
                        Ongoing => {
                            let mut nprobs = 0.0;
                            let mut total = 0.0;
                            for (dice, n) in ALL_21 {
                                let (_, v) = self.nply(depth - 1, !maxer, p, &dice);
                                total += v;
                                nprobs += n;
                            }
                            (*p, total / nprobs)
                        }
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
            }
        } else {
            if maxer {
                pos.possible_positions(dice)
                    .iter()
                    .map(|p| match p.game_state() {
                        GameOver(result) => (
                            *p,
                            if !pos.turn {
                                self.result_value(result)
                            } else {
                                1.0 - self.result_value(result)
                            },
                        ), // check p.turn
                        Ongoing => {
                            let mut nprobs = 0.0;
                            let mut total = 0.0;
                            for (dice, n) in ALL_21 {
                                let (_, v) = self.nply(depth - 1, !maxer, p, &dice);
                                total += v;
                                nprobs += n;
                            }
                            (*p, total / nprobs)
                        }
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
            } else {
                pos.possible_positions(dice)
                    .iter()
                    .map(|p| match p.game_state() {
                        GameOver(result) => (
                            *p,
                            if !pos.turn {
                                self.result_value(result)
                            } else {
                                1.0 - self.result_value(result)
                            },
                        ), // check p.turn
                        Ongoing => {
                            let mut nprobs = 0.0;
                            let mut total = 0.0;
                            for (dice, n) in ALL_21 {
                                let (_, v) = self.nply(depth - 1, !maxer, p, &dice);
                                total += v;
                                nprobs += n;
                            }
                            (*p, total / nprobs)
                        }
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
            }
        }
    }
}

impl<G: State + Send, B: Backend> Evaluator<FState<G>> for TDModel<B> {
    fn best_position(&self, pos: &FState<G>, dice: &Dice) -> FState<G> {
        self.nply(2, true, pos, dice).0
    }

    // fn best_position(&self, pos: &FState<G>, dice: &Dice) -> FState<G> {
    //     let device = B::Device::default();

    //     let positions = pos.possible_positions(dice);

    //     if pos.turn {
    //         let inputs =
    //             self.input_tensor(&device, positions.iter().map(|p| p.position()).collect());
    //         let outputs = self.forward(inputs);

    //         let data: Data<f32, 2> = outputs.into_data().convert();

    //         positions
    //             .iter()
    //             .enumerate()
    //             .map(|(i, pos)| (*pos, data.value[i]))
    //             .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    //             .unwrap()
    //             .0
    //     } else {
    //         let inputs = self.input_tensor(
    //             &device,
    //             positions.iter().map(|p| p.position().flip()).collect(),
    //         );
    //         let outputs = self.forward(inputs);

    //         let data: Data<f32, 2> = outputs.into_data().convert();

    //         positions
    //             .iter()
    //             .enumerate()
    //             .map(|(i, pos)| (*pos, data.value[i]))
    //             .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    //             .unwrap()
    //             .0
    //     }
    // }
}
