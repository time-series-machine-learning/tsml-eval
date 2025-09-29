import os
from collections import OrderedDict

import numpy as np
import wandb
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from data_preprocess import create_dataLoader2
from model_utils import resample_from_normal
from network import HiddenLayerMLP
import torch


class CriticFunc(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super().__init__()
        cat_dim = x_dim + y_dim
        self.critic = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cat_dim, cat_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(cat_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        cat = torch.cat((x, y), dim=-1)
        return self.critic(cat)


class DisentangledMILoss(nn.Module):
    def __init__(self, x_dim, y_dim, dropout=0.1):
        super().__init__()
        self.critic_st = CriticFunc(x_dim, y_dim, dropout)

    def forward(self, z_c, z_nc):
        # I(Z_c, Z_nc)
        idx = torch.randperm(z_nc.shape[0])
        zc_shuffle = z_nc[idx].view(z_nc.size())
        f_cnc = self.critic_st(z_c, z_nc)
        f_c_nc = self.critic_st(z_c, zc_shuffle)
        mubo = f_cnc - f_c_nc
        pos_mask = torch.zeros_like(f_cnc)
        pos_mask[mubo < 0] = 1
        mubo_mask = mubo * pos_mask
        reg = (mubo_mask ** 2).mean()
        return mubo.mean() + reg


class HiddenNetworksOPtions(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 dropout_list=None,
                 model_type='encoder'):
        super().__init__()
        self.model_type = model_type
        self.hidden_net = HiddenLayerMLP(input_dim=input_dim, hidden_dim=hidden_dim, model_type=model_type,
                                         dropout_list=dropout_list)

    def forward(self, x):
        return self.hidden_net(x)


class VAE(nn.Module):
    def __init__(self,
                 feat_dim,
                 seq_len,
                 latent_dim,
                 hidden_dim,
                 dropout_list
                 ):
        super().__init__()

        self.feat_dim = feat_dim
        self.seq_len = seq_len

        self.encoder_network = HiddenNetworksOPtions(input_dim=feat_dim * seq_len,
                                                     hidden_dim=hidden_dim,
                                                     dropout_list=dropout_list,
                                                     model_type='encoder')

        self.encoder_hidden = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim

        self.mean_net = nn.ModuleList([nn.Linear(self.encoder_hidden, latent_dim // 2) for _ in range(2)])
        self.var_net = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.encoder_hidden, latent_dim // 2), nn.Softplus()) for _ in
             range(2)])

        self.output = nn.Sequential(
            HiddenNetworksOPtions(input_dim=latent_dim, hidden_dim=hidden_dim, dropout_list=dropout_list,
                                  model_type='decoder'),
            nn.Linear(hidden_dim[0] if isinstance(hidden_dim, list) else hidden_dim, feat_dim * seq_len))

    def decoder_output(self, z):
        output = self.output(z)
        output = output.view(output.size(0), self.feat_dim, self.seq_len)
        return output

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder_network(x)
        mean = [fc_mu(x) for fc_mu in self.mean_net]
        var = [fc_var(x) for fc_var in self.var_net]
        qz = [resample_from_normal(mean[i], var[i]) for i in range(2)]
        return qz, mean, var


class CFAMGVAE(nn.Module):
    def __init__(self,
                 feat_dim,
                 seq_len,
                 latent_dim,
                 hidden_dim,
                 dropout_list
                 ):
        super().__init__()

        self.vae = VAE(feat_dim, seq_len, latent_dim, hidden_dim, dropout_list)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 2),
            nn.ReLU(),
            nn.Softmax(dim=1))

        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.mse = nn.MSELoss()

        self.MINet = DisentangledMILoss(latent_dim // 2, latent_dim // 2)

    def compute_KL2(self, z_q_mean, z_q_var):
        z_q_var = torch.clamp(z_q_var, min=1e-8)
        kl_divergence = -0.5 * torch.sum(1 + torch.log(z_q_var) - z_q_mean ** 2 - z_q_var, dim=-1)
        return torch.mean(kl_divergence)

    def compute_classifier_loss(self, input, label):
        pred_label = self.classifier(input)
        if label.dtype != torch.long:
            label = label.long()
        loss = self.criterion(pred_label, label)
        return loss.mean()

    def swap_classifier_loss(self, sample10, sample11, sample21, label):
        # swap posority class loss, label: pos
        swap_index = np.random.permutation(sample11.size(0))
        swap = torch.cat([sample10.detach().clone(), sample11[swap_index].detach().clone()], dim=1)
        swap_loss1 = self.compute_classifier_loss(swap, label)
        # concat causal pos and non-causal neg loss, label: pos
        c_nc_cat = torch.cat([sample10.detach().clone(), sample21.detach().clone()], dim=1)
        swap_pos_loss2 = self.compute_classifier_loss(c_nc_cat, label)
        loss = swap_loss1 + swap_pos_loss2
        return loss

    def computer_loss(self, mu_list, var_list, z_sample, x, output):
        # computer KL Loss
        kl_loss = 0.0
        count = 0
        for mu, var in zip([*mu_list], [*var_list]):
            kl_loss += self.compute_KL2(mu, var)
            count += 1
        kl_loss = kl_loss / count

        # computer reconstruction loss
        reconstruction_loss = self.mse(output, x)

        mi_loss = self.MINet(z_sample[0].detach(), z_sample[1].detach())
        loss = (kl_loss, reconstruction_loss, mi_loss)
        return loss

    def forward(self, x, label=None, return_loss=True):
        qz, mean, var = self.vae(x)

        if return_loss:
            output = self.vae.decoder_output(torch.concat(qz, dim=1))
            loss = self.computer_loss(mean, var, qz, x, output)
            return qz, loss
        else:
            return qz


class CFAMG:
    def __init__(self, args):
        self.args = args

        self.device = args.device
        project_name = args.project_name

        if args.wandb:
            wandb.init(project=project_name, config=args)

        if args.tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f"Experiment/{project_name}/VAE/{args.exp_name}")
            self.writer_cls = SummaryWriter(f"Experiment/{project_name}/Classifier/{args.exp_name}")

        self.log_dir_path = os.path.join(args.log_dir, project_name, args.dataset_name)
        self.result_dir = os.path.join(self.log_dir_path, "result")
        os.makedirs(self.result_dir, exist_ok=True)
        self.dataset = args.dataset
        self.pos_dataloader, self.neg_dataloader, self.feat_dim, self.seq_len, self.batch_size = create_dataLoader2(
            args.dataset,
            args.batch_size)

        self.pos_model = CFAMGVAE(feat_dim=self.feat_dim,
                                  seq_len=self.seq_len,
                                  latent_dim=self.args.latent_dim,
                                  hidden_dim=self.args.hidden_dim,
                                  dropout_list=self.args.dropout_list).to(self.device)
        self.neg_model = CFAMGVAE(feat_dim=self.feat_dim,
                                  seq_len=self.seq_len,
                                  latent_dim=self.args.latent_dim,
                                  hidden_dim=self.args.hidden_dim,
                                  dropout_list=self.args.dropout_list).to(self.device)
        self.optimizer_pos = torch.optim.Adam(self.pos_model.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)
        self.optimizer_neg = torch.optim.Adam(self.neg_model.parameters(), lr=self.args.lr,
                                              weight_decay=self.args.weight_decay)

    def train_on_data(self):
        print('*' * 50)
        print('Main Training Starts ...')
        print('*' * 50)

        if self.args.use_lr_decay:
            self.scheduler_pos = torch.optim.lr_scheduler.StepLR(self.optimizer_pos, step_size=self.args.lr_decay_step,
                                                                 gamma=self.args.lr_gamma)
            self.scheduler_neg = torch.optim.lr_scheduler.StepLR(self.optimizer_neg, step_size=self.args.lr_decay_step,
                                                                 gamma=self.args.lr_gamma)

        best_loss = float('inf')
        patience = 10
        counter = 0
        for epoch in tqdm(range(self.args.num_epochs)):
            self.pos_model.train()
            self.neg_model.train()

            if epoch < 30:
                for param in self.pos_model.parameters():
                    param.requires_grad = True
                lr_scale = 1.0
            else:
                for param in self.pos_model.vae.encoder_network.parameters():
                    param.requires_grad = False
                lr_scale = 0.1

            for param_group in self.optimizer_pos.param_groups:
                param_group['lr'] = self.args.lr * lr_scale

            from itertools import cycle
            pos_loader_cycle = cycle(self.pos_dataloader)
            neg_loader_iter = iter(self.neg_dataloader)

            self.pos_kl_loss, self.pos_reconstruction_loss, self.pos_mi_loss, self.pos_swap_loss = [], [], [], []
            self.neg_kl_loss, self.neg_reconstruction_loss, self.neg_mi_loss, self.neg_swap_loss = [], [], [], []
            self.total_pos_loss, self.total_neg_loss = [], []
            for _ in range(len(self.neg_dataloader)):
                pos_samp, pos_label = next(pos_loader_cycle)
                neg_samp, neg_label = next(neg_loader_iter)

                # 数据增强与设备迁移
                pos_samp, pos_label = pos_samp.to(self.device), pos_label.to(self.device)
                neg_samp, neg_label = neg_samp.to(self.device), neg_label.to(self.device)

                # 前向传播
                pos_z, pos_losses = self.pos_model(pos_samp, pos_label)
                pos_kl_loss, pos_reconstruction_loss, pos_mi_loss = pos_losses
                self.pos_kl_loss.append(pos_kl_loss)
                self.pos_reconstruction_loss.append(pos_reconstruction_loss)
                self.pos_mi_loss.append(pos_mi_loss)
                neg_z, neg_losses = self.neg_model(neg_samp, neg_label)
                neg_kl_loss, neg_reconstruction_loss, neg_mi_loss = neg_losses
                self.neg_kl_loss.append(neg_kl_loss)
                self.neg_reconstruction_loss.append(neg_reconstruction_loss)
                self.neg_mi_loss.append(neg_mi_loss)

                # 对齐批次并计算Swap Loss（带正则化）
                pos_z1, pos_z2 = pos_z
                neg_z1, neg_z2 = neg_z
                pos_swap_loss = self.pos_model.swap_classifier_loss(pos_z1, pos_z2, neg_z2, pos_label)
                neg_swap_loss = self.neg_model.swap_classifier_loss(neg_z1, neg_z2, pos_z2, neg_label)
                self.pos_swap_loss.append(pos_swap_loss)
                self.neg_swap_loss.append(neg_swap_loss)

                # 加权联合损失
                total_pos_loss = pos_kl_loss + pos_reconstruction_loss + self.args.w_lambda * pos_mi_loss + self.args.w_beta * pos_swap_loss
                total_neg_loss = neg_kl_loss + neg_reconstruction_loss + self.args.w_lambda * neg_mi_loss + self.args.w_beta * neg_swap_loss
                self.total_pos_loss.append(total_pos_loss)
                self.total_neg_loss.append(total_neg_loss)

                # 梯度计算与裁剪
                self.optimizer_pos.zero_grad()
                self.optimizer_neg.zero_grad()
                total_pos_loss.backward(retain_graph=True)
                total_neg_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.pos_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.neg_model.parameters(), max_norm=1.0)

                # 选择性更新分类器
                # self.optimizer_classifier_pos.step()
                self.optimizer_pos.step()
                # self.optimizer_classifier_neg.step()
                self.optimizer_neg.step()

            if self.args.use_lr_decay:
                self.scheduler_pos.step()
                self.scheduler_neg.step()

            if epoch % self.args.save_freq == 0:
                self.save_model(epoch)

            if epoch % self.args.log_freq == 0:
                self.board_loss(epoch)

            current_reconstruction_loss = sum(self.total_pos_loss) / len(self.total_pos_loss)

            # 早停与模型保存
            if current_reconstruction_loss < best_loss:
                best_loss = current_reconstruction_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.save_model(epoch)
                    break

    def generator_sample(self):
        self.pos_model.eval()
        self.neg_model.eval()
        X_pos, y_pos = self.args.dataset["train_data_pos"]
        X_neg, y_neg = self.args.dataset["train_data_neg"]
        num_majority, num_minority = len(y_neg), len(y_pos)
        num_diff_samp = num_majority - num_minority
        generated_samples = []

        with torch.no_grad():
            for pos_samp, _ in self.pos_dataloader:
                pos_samp = pos_samp.squeeze(0).to(self.device)
                z_pos = self.pos_model(pos_samp, return_loss=False)[0]
                for neg_samp, _ in self.neg_dataloader:
                    neg_samp = neg_samp.squeeze(0).to(self.device)
                    z_neg = self.neg_model(neg_samp, return_loss=False)[1]
                    if z_neg.shape[0] < z_pos.shape[0]:
                        z_pos = z_pos[:z_neg.shape[0]]
                    elif z_neg.shape[0] > z_pos.shape[0]:
                        z_neg = z_neg[:z_pos.shape[0]]
                    z = torch.concat((z_pos, z_neg), dim=-1)
                    generated_samp = self.pos_model.vae.decoder_output(z)
                    generated_samples.append(generated_samp)

        generated_samples = torch.cat(generated_samples, dim=0)
        if generated_samples.size(0) < num_diff_samp:
            num_diff_samp = generated_samples.size(0)
        generated_samples = generated_samples[:num_diff_samp].cpu().detach().numpy()
        balance_pos_samp = np.concatenate((X_pos, generated_samples), axis=0)
        balance_pos_label = np.concatenate((y_pos, np.ones((num_diff_samp,))), axis=0)

        balance_samp = np.concatenate((balance_pos_samp, X_neg), axis=0)
        balance_label = np.concatenate((balance_pos_label, y_neg), axis=0)

        return balance_samp, balance_label, generated_samples

    def save_model(self, epoch, model_name='CFAMG'):
        model_path = os.path.join(self.result_dir, f"model_{model_name}_{epoch}.pth")
        state_dict = {
            'steps': epoch,
            'pos_state_dict': self.pos_model.state_dict(),
            'pos_optimizer': self.optimizer_pos.state_dict(),
            'neg_state_dict': self.neg_model.state_dict(),
            'neg_optimizer': self.optimizer_neg.state_dict()
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'\n {epoch} model saved ...')

    def board_loss(self, epoch):
        print('\n')
        n1, n2 = len(self.pos_dataloader), len(self.neg_dataloader)
        print(
            f"Epoch : {epoch},"
            f" Pos Loss : {sum(self.total_pos_loss) / n2},"
            f" Pos KL Loss : {sum(self.pos_kl_loss) / n2}"
            f" Pos Recon Loss : {sum(self.pos_reconstruction_loss) / n2}"
            f" Pos Swap Loss : {self.args.w_beta * sum(self.pos_swap_loss) / n2}"
            f" Pos MI Loss : {self.args.w_lambda * sum(self.pos_mi_loss) / n2}"
        )
        print(
            f"Epoch : {epoch},"
            f" Neg Loss : {sum(self.total_neg_loss) / n2},"
            f" Neg KL Loss : {sum(self.neg_kl_loss) / n2}"
            f" Neg Recon Loss : {sum(self.neg_reconstruction_loss) / n2}"
            f" Neg Swap Loss : {self.args.w_beta * sum(self.neg_swap_loss) / n2}"
            f" Neg MI Loss : {self.args.w_lambda * sum(self.neg_mi_loss) / n2}"
        )

    def load_model(self, model_path):
        model_param = torch.load(model_path)
        self.pos_model.load_state_dict(model_param["pos_state_dict"])
        self.pos_model.eval()
        self.neg_model.load_state_dict(model_param["neg_state_dict"])
        self.neg_model.eval()
