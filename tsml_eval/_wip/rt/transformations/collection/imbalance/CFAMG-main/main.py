import os
import argparse
import numpy as np
import pandas as pd
import torch
from data_preprocess import set_seed, load_dataset
from cfamg import CFAMG

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)


def parse_args():
    parser = argparse.ArgumentParser(description='CFAMG')

    # Set path
    parser.add_argument('--output_dir', type=str, default='CFAMG', help='directory to save the results')

    # Set data parameter
    parser.add_argument('--batch_size', type=int, default=32, help='data batch size')

    # Set logging
    parser.add_argument("--log_dir", default='./Exp_log', type=str, help='path for saving model')
    parser.add_argument("--data_dir", action='store_true', help='path for loading data')
    parser.add_argument("--log_freq", default=5, type=int, help='frequency to log on tensorboard')
    parser.add_argument("--save_freq", default=200, type=int, help='frequency to save model checkpoint')
    parser.add_argument("--tensorboard", action="store_true", help="whether to use tensorboard")
    parser.add_argument("--wandb", action='store_true', help='whether to use wandb')

    # Set train parameter
    parser.add_argument('--num_epochs', type=int, default=201, help='max iters for training')
    parser.add_argument('--latent_dim', type=int, default=64, help='latent dim')
    parser.add_argument('--hidden_dim', type=int, default=[32, 64, 128],
                        help='hidden layer dimension')
    parser.add_argument("--dropout_list", type=int, default=[0.1, 0.1, 0.2], help='MLP Dropout')
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="whether to use learning rate decay")
    parser.add_argument("--lr_decay_step", help="learning rate decay steps", type=int, default=100)
    parser.add_argument("--lr_gamma", help="lr gamma", type=float, default=0.1)
    parser.add_argument("--lr", help='learning rate', default=1e-3, type=float)
    parser.add_argument("--weight_decay", help='weight_decay', default=1e-4, type=float)
    parser.add_argument("--temp_epochs", help="curriculum steps", type=int, default=300)
    parser.add_argument('--cls_num_epochs', type=int, default=100, help='max iters for classifier training')
    parser.add_argument('--beta', type=float, default=10, help='VAE parameter')

    args = parser.parse_args()
    args.save_path = os.path.join(os.getcwd(), args.output_dir)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
