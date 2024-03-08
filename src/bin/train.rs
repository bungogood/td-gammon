use std::path::PathBuf;

use burn::module::Module;
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Model path
    #[arg(short = 'm', long = "model")]
    model_path: Option<PathBuf>,

    /// Directory
    #[arg(short = 'd', long = "dir")]
    dir: Option<PathBuf>,

    /// Verbose
    #[arg(short = 'v', long = "verbose", default_value = "false")]
    verbose: bool,

    /// Use CPU only
    #[arg(short = 'c', long = "cpu", default_value = "false")]
    cpu_only: bool,
}

use bkgm::{Backgammon, Hypergammon};
use burn::backend::libtorch::{LibTorch, LibTorchDevice};
use burn::backend::Autodiff;
use burn::record::NoStdTrainingRecorder;
use td_gammon::model::{ModelConfig, TDModel};
use td_gammon::train::{TDConfig, TDTrainer};

fn get_device(cup_only: bool) -> LibTorchDevice {
    if cup_only {
        LibTorchDevice::Cpu
    } else {
        #[cfg(not(target_os = "macos"))]
        let device = LibTorchDevice::Cuda(0);
        // MacOs Mps too slow
        #[cfg(target_os = "macos")]
        let device = LibTorchDevice::Cpu;
        // let device = LibTorchDevice::Mps;
        device
    }
}

pub fn run(args: &Args) {
    let device = get_device(args.cpu_only);

    let config = ModelConfig::new()
        // .with_layers(1)
        .with_neurons(160)
        .with_nply(1);

    let td_config = TDConfig::new().with_learning_rate(0.1).with_td_decay(0.7);

    let model = match &args.model_path {
        Some(path) => TDModel::<Autodiff<LibTorch>>::init_with(config, device, path),
        None => TDModel::<Autodiff<LibTorch>>::new(config, &device),
    };

    let mut td: TDTrainer<Autodiff<LibTorch>> = TDTrainer::new(device.clone(), td_config);

    let model = td.train::<Hypergammon>(args.dir.clone(), model, 500_000);

    // model
    //     .save_file(format!("model/td-next"), &NoStdTrainingRecorder::new())
    //     .expect("Failed to save trained model");
}

fn main() {
    let args = Args::parse();
    run(&args);
}
