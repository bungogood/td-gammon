use std::path::PathBuf;

use bkgm::Hypergammon;
use burn::{backend::LibTorch, tensor::backend::Backend};
use td_gammon::{
    duel::duel,
    evaluator::{Evaluator, HyperEvaluator, PubEval, RandomEvaluator},
    fstate::FState,
    model::{ModelConfig, TDModel},
    train::TDConfig,
};

fn run<B: Backend>(config: ModelConfig, td_config: TDConfig) {
    // let opponent: Box<dyn Evaluator<FState<Hypergammon>>> = match opponent {
    //     "random" => RandomEvaluator::new(),
    //     "hyper" => HyperEvaluator::new().unwrap(),
    //     "pub" => PubEval::new(),
    //     _ => panic!("Unknown opponent"),
    // };

    let oponent = PubEval::new();

    let device = B::Device::default();
    println!("round,equity,win,win_n,win_g,win_b,lose_n,lose_g,lose_b");
    let base = format!(
        "model/expr/{}-ply-{}-{}-{}-{}",
        config.nply,
        config.layers,
        config.neurons,
        td_config.learning_rate.to_string().replace(".", ""),
        td_config.td_decay.to_string().replace(".", ""),
    );
    for num in 0..=1000 {
        let round = num * 1000;
        let path = PathBuf::from(format!("{}/games-{}.bin", base, round));
        let eval = TDModel::<B>::init_with(config.clone(), device.clone(), &path);
        let probs = duel::<FState<Hypergammon>>(eval, oponent, 10000);
        println!(
            "{},{:.3},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            round,
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
}

fn main() {
    let config = ModelConfig::new()
        // .with_layers(1)
        .with_neurons(160)
        .with_nply(1);

    let td_config = TDConfig::new().with_learning_rate(0.1).with_td_decay(0.7);
    run::<LibTorch>(config, td_config);
}
