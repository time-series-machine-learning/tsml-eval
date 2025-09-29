import os
import numpy as np
import pandas as pd
import torch
from tsml_eval._wip.rt.transformations.collection.imbalance.pk_cfamg.data_preprocess import set_seed, load_dataset
from tsml_eval._wip.rt.transformations.collection.imbalance.pk_cfamg.cfamg import CFAMG

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)


def parse_args():
    # Set path
    output_dir = 'CFAMG'
    log_dir = './Exp_log'
    data_dir = None  # action='store_true' implies a boolean flag, so default to None or False

    # Set data parameter
    batch_size = 32

    # Set logging
    log_freq = 5
    save_freq = 200
    tensorboard = False  # action="store_true" implies a boolean flag
    wandb = False  # action='store_true' implies a boolean flag

    # Set train parameter
    num_epochs = 201
    latent_dim = 64
    hidden_dim = [32, 64, 128]
    dropout_list = [0.1, 0.1, 0.2]
    use_lr_decay = False
    lr_decay_step = 100
    lr_gamma = 0.1
    lr = 1e-3
    weight_decay = 1e-4
    temp_epochs = 300
    cls_num_epochs = 100
    beta = 10

    # Create a class or object to hold the parameters
    class Args:
        pass

    args = Args()
    args.output_dir = output_dir
    args.log_dir = log_dir
    args.data_dir = data_dir
    args.batch_size = batch_size
    args.log_freq = log_freq
    args.save_freq = save_freq
    args.tensorboard = tensorboard
    args.wandb = wandb
    args.num_epochs = num_epochs
    args.latent_dim = latent_dim
    args.hidden_dim = hidden_dim
    args.dropout_list = dropout_list
    args.use_lr_decay = use_lr_decay
    args.lr_decay_step = lr_decay_step
    args.lr_gamma = lr_gamma
    args.lr = lr
    args.weight_decay = weight_decay
    args.temp_epochs = temp_epochs
    args.cls_num_epochs = cls_num_epochs
    args.beta = beta

    args.save_path = os.path.join(os.getcwd(), args.output_dir)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', args.device)
    return args


def run_CFAMG(args):
    model = CFAMG(args)
    model.train_on_data()
    X_train, y_train, _ = model.generator_sample()
    data_save_path = os.path.join(os.getcwd(), args.save_file, args.project_name, args.dataset_name)
    os.makedirs(data_save_path, exist_ok=True)
    np.save(os.path.join(data_save_path, 'X_train_syn_sample.npy'), X_train)
    np.save(os.path.join(data_save_path, 'y_train_syn_sample.npy'), y_train)
    print('Synthetic data has been saved ! ')


if __name__ == "__main__":
    seed = 2024
    set_seed(seed)
    args = parse_args()
    datasets_path = os.getcwd()
    time_log = pd.DataFrame(columns=["Datasets", "Time(s)"])
    args.save_file = "synthetic_dataset"
    tsu_name = 'FiftyWords'
    args.w_lambda, args.w_beta = 1, 1
    args.project_name = f'CFAMG'
    args.exp_name = tsu_name
    args.dataset_name = tsu_name
    print(f"Model: {args.exp_name} || Dataset : {tsu_name}")
    dataset, ir = load_dataset(dataset_name=tsu_name, root_path=datasets_path)
    args.dataset = dataset
    run_CFAMG(args)
