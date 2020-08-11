import torch
import torch.nn as nn
import pandas as pd
from .dataset import EncoderTrainingDataset, DecoderTrainingDataset, TestingDataset
from torch.utils.data import DataLoader
import torch.optim as optimizer
import itertools
import time


class TripletNetwork(nn.Module):
    def __init__(self, subnet):
        super().__init__()
        self.subnet = subnet

    def forward(self, anchor, positive, negative):
        latent_anchor = self.subnet(anchor)
        latent_positive = self.subnet(positive)
        latent_negative = self.subnet(negative)
        dist_positive = nn.functional.pairwise_distance(latent_anchor,
                                                        latent_positive)
        dist_negative = nn.functional.pairwise_distance(latent_anchor,
                                                        latent_negative)
        return dist_positive, dist_negative, \
            latent_anchor, latent_positive, latent_negative


class Encoder(nn.Module):
    def __init__(
            self,
            gene_num: int,
            latent_size: int = 10,
            dropout_rate: float = 0.0):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(gene_num, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1000, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(200, latent_size)
        )
        self.coefficient = torch.nn.Parameter(torch.ones(gene_num))

    def forward(self, network_input):
        network_input = self.coefficient * network_input
        output = self.layer3(self.layer2(self.layer1(network_input)))
        return output


class DecoderR(nn.Module):
    def __init__(self, gene_num: int, latent_size: int = 10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_size, gene_num),
            nn.ReLU(inplace=True)
        )

    def forward(self, network_input):
        output = self.layer1(network_input)
        return output


class DecoderD(nn.Module):
    def __init__(
            self,
            gene_num: int,
            one_hot_num: int,
            latent_size: int = 10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_size + one_hot_num, 200),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(200, 1000),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(1000, gene_num)
        )

    def forward(self, network_input):
        output = self.layer3(self.layer2(self.layer1(network_input)))
        return output


class SCAT(nn.Module):
    def __init__(
            self,
            data: pd.Series,
            metadata: pd.Series,
            latent_size: int = 10,
            dropout_rate: float = 0.0,
            learning_rate: float = 5e-4,
            num_workers: int = 0,
            batch_size: int = 256,
            use_gpu: bool = True):
        super().__init__()
        gene_num = len(data)
        self.one_hot_num = len(set(metadata['batch']))

        self.use_device = torch.device("cpu")
        if use_gpu:
            if torch.cuda.is_available():
                self.use_device = torch.device("cuda")
            else:
                use_gpu = False

        self.encoder_training_dataset = EncoderTrainingDataset(
            data=data, metadata=metadata, anchor_score=1.0)
        self.decoder_training_dataset = DecoderTrainingDataset(
            dataset=self.encoder_training_dataset)
        self.encoder_training_dataloader = DataLoader(
            self.encoder_training_dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=use_gpu)
        self.decoder_training_dataloader = DataLoader(
            self.decoder_training_dataset,
            shuffle=True,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=use_gpu)

        self.encoder = Encoder(
            gene_num=gene_num,
            latent_size=latent_size,
            dropout_rate=dropout_rate)
        self.decoder_r = DecoderR(
            gene_num=gene_num,
            latent_size=latent_size)
        self.decoder_d = DecoderD(
            gene_num=gene_num,
            latent_size=latent_size,
            one_hot_num=self.one_hot_num)

        self.set_device(self.use_device)

        self.encoder_triplet_network = TripletNetwork(subnet=self.encoder)
        self.decoder_d_triplet_network = TripletNetwork(subnet=self.decoder_d)
        self.decoder_r_triplet_network = TripletNetwork(subnet=self.decoder_r)

        self.first_stage_optimizer = optimizer.Adam(
            self.encoder.parameters(), lr=learning_rate)
        self.second_stage_optimizer = optimizer.Adam(
            itertools.chain(
                self.decoder_d.parameters(),
                self.decoder_r.parameters()),
            lr=learning_rate)

    def forward(self, network_input):
        output = self.decoder_r(self.encoder(network_input))
        return output

    def set_device(self, use_device: torch.device = "cpu"):
        self.use_device = use_device
        self.encoder = self.encoder.to(use_device)
        self.decoder_r = self.decoder_r.to(use_device)
        self.decoder_d = self.decoder_d.to(use_device)

    def train(self, epochs: int = 50):
        epoch_start_time = time.time()
        for epoch_i in range(epochs):
            train_loss = self.one_epoch_encoder()
            progress = ('#' * int(float(epoch_i) /
                                  epochs * 30 + 1)).ljust(30)
            print(
                'Stage 1: [ %03d / %03d ] %6.2f sec(s) | %s | Train Loss: %3.6f' %
                (epoch_i + 1, epochs, (time.time() - epoch_start_time), progress, train_loss),
                end='\r', flush=True)
        print("\n")

        epoch_start_time = time.time()
        for epoch_i in range(epochs):
            train_loss = self.one_epoch_decoder()
            progress = ('#' * int(float(epoch_i) /
                                  epochs * 30 + 1)).ljust(30)
            print(
                'Stage 2: [ %03d / %03d ] %6.2f sec(s) | %s | Train Loss: %3.6f' %
                (epoch_i + 1, epochs, (time.time() - epoch_start_time), progress, train_loss),
                end='\r', flush=True)
        print("\nTraining finish!\n")

    def one_epoch_encoder(self) -> float:
        for _, data in enumerate(self.encoder_training_dataloader, 0):
            cell, positive_cell, negative_cell, confidence = data
            train_loss = self.encoder_loss(
                cell, positive_cell, negative_cell, confidence)
            self.first_stage_optimizer.zero_grad()
            train_loss.backward()
            self.first_stage_optimizer.step()
        return train_loss.detach()

    def one_epoch_decoder(self) -> float:
        one_hot = torch.eye(self.one_hot_num).requires_grad_(True)
        labels = self.decoder_training_dataset.label_code
        for _, data in enumerate(self.decoder_training_dataloader, 0):
            cell, positive_cell, negative_cell, idx, positive_idx, negative_idx = data
            train_loss = self.decoder_loss(cell,
                                           positive_cell,
                                           negative_cell,
                                           one_hot[labels[idx], ],
                                           one_hot[labels[positive_idx], ],
                                           one_hot[labels[negative_idx], ])
            self.second_stage_optimizer.zero_grad()
            train_loss.backward()
            self.second_stage_optimizer.step()
        return train_loss.detach()

    def encoder_loss(self, cell, positive_cell, negative_cell, confidence):
        zero = torch.Tensor([0.0]).to(self.use_device)
        cell, positive_cell, negative_cell, confidence = \
            cell.to(self.use_device), \
            positive_cell.to(self.use_device), \
            negative_cell.to(self.use_device), \
            confidence.to(self.use_device)

        dist_positive, dist_negative, _, _, _ = \
            self.encoder_triplet_network(cell, positive_cell, negative_cell)
        train_loss = torch.mean(
            torch.max(dist_positive - dist_negative + confidence, zero))
        return train_loss

    def decoder_loss(
            self,
            ori_cell,
            ori_positive_cell,
            ori_negative_cell,
            one_hot_cell,
            one_hot_positive_cell,
            one_hot_negative_cell):
        ori_cell, ori_positive_cell, ori_negative_cell = \
            ori_cell.to(self.use_device), \
            ori_positive_cell.to(self.use_device), \
            ori_negative_cell.to(self.use_device)
        one_hot_cell, one_hot_positive_cell, one_hot_negative_cell = \
            one_hot_cell.to(self.use_device), \
            one_hot_positive_cell.to(self.use_device), \
            one_hot_negative_cell.to(self.use_device)
        target = torch.FloatTensor(
            len(ori_cell)).fill_(-1).requires_grad_(True).to(self.use_device)

        _, _, encode_ori_cell, encode_ori_positive_cell, encode_ori_negative_cell = \
            self.encoder_triplet_network(ori_cell,
                                         ori_positive_cell,
                                         ori_negative_cell)

        one_hot_cell = torch.cat(
            (encode_ori_cell, one_hot_cell), 1)
        one_hot_positive_cell = torch.cat(
            (encode_ori_positive_cell, one_hot_positive_cell), 1)
        one_hot_negative_cell = torch.cat(
            (encode_ori_negative_cell, one_hot_negative_cell), 1)

        triplet_loss = nn.MarginRankingLoss(margin=0.5)
        mse_loss = nn.MSELoss()

        _, _, recon_r_cell, recon_r_positive_cell, recon_r_negative_cell = \
            self.decoder_r_triplet_network(encode_ori_cell,
                                           encode_ori_positive_cell,
                                           encode_ori_negative_cell)
        dist_positive, dist_negative, recon_d_cell, recon_d_positive_cell, recon_d_negative_cell = \
            self.decoder_d_triplet_network(one_hot_cell,
                                           one_hot_positive_cell,
                                           one_hot_negative_cell)
        recon, positive_recon, negative_recon = recon_r_cell + recon_d_cell, \
            recon_r_positive_cell + recon_d_positive_cell, \
            recon_r_negative_cell + recon_d_negative_cell

        train_loss = triplet_loss(dist_positive, dist_negative, target) + \
            (mse_loss(recon, ori_cell) +
             mse_loss(positive_recon, ori_positive_cell) +
             mse_loss(negative_recon, ori_negative_cell)) / 3
        return train_loss


def evaluation(
        model: SCAT,
        data: pd.Series,
        metadata: pd.Series,
        num_workers: int = 0,
        batch_size: int = 256,
        use_gpu: bool = False):

    device = torch.device("cpu")
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            use_gpu = False
    model.set_device(device)

    testing_dataset = TestingDataset(data=data, metadata=metadata)
    testing_dataloader = DataLoader(
        testing_dataset,
        shuffle=False,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=use_gpu)

    recon = torch.Tensor(0).to(device)
    for _, cell in enumerate(testing_dataloader, 0):
        cell = cell.to(device)
        output = model(cell)
        recon = torch.cat([recon, output], 0)
    return recon.cpu().detach().numpy()
