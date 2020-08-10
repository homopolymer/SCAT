import torch
import torch.nn as nn
import pandas as pd
from .dataset import EncoderTrainingDataset, DecoderTrainingDataset
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
    def __init__(self, gene_num, latent_size=10, dropout_rate=0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(gene_num, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(100, latent_size)
        )

    def forward(self, network_input):
        output = self.layer3(self.layer2(self.layer1(network_input)))
        return output


class DecoderR(nn.Module):
    def __init__(self, gene_num: int, latent_size: int = 10, dropout_rate=0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_size, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(100, gene_num),
            nn.ReLU(inplace=True)
        )

    def forward(self, network_input):
        output = self.layer2(self.layer1(network_input))
        return output


class DecoderD(nn.Module):
    def __init__(
            self,
            gene_num: int,
            one_hot_num: int,
            latent_size: int = 10,
            dropout_rate: float = 0.3):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_size + one_hot_num, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.layer2 = nn.Sequential(
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
            dropout_rate: float = 0.3,
            learning_rate: float = 1e-3,
            num_workers: int = 0,
            batch_size: int = 256,
            use_gpu: bool = True):
        super().__init__()
        gene_num = len(data)
        one_hot_num = len(set(metadata['batch']))

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
            latent_size=latent_size,
            dropout_rate=dropout_rate)
        self.decoder_d = DecoderD(
            gene_num=gene_num,
            latent_size=latent_size,
            one_hot_num=one_hot_num,
            dropout_rate=dropout_rate)

        self.encoder_triplet_network = TripletNetwork(subnet=self.encoder)
        self.decoder_triplet_network = TripletNetwork(subnet=self.decoder_d)

        self.first_stage_optimizer = optimizer.Adam(
            itertools.chain(
                self.encoder.parameters(),
                self.decoder_d.parameters(),
                self.decoder_r.parameters()),
            lr=learning_rate)
        self.second_stage_optimizer = optimizer.Adam(
            itertools.chain(
                self.decoder_d.parameters(),
                self.decoder_r.parameters()),
            lr=learning_rate)

    def forward(self, network_input):
        output = self.decoder_r(self.encoder(network_input))
        return output

    def set_mode(self, mode: int = 0):
        """
        0: first training stage
        1: second training stage
        others: test stage
        """
        if mode == 0:
            self.encoder.train()
            self.decoder_d.train()
            self.decoder_r.train()
        elif mode == 1:
            self.encoder.eval()
            self.decoder_d.train()
            self.decoder_r.train()
        else:
            self.encoder.eval()
            self.decoder_d.eval()
            self.decoder_r.eval()

    def train(self, epochs: int = 50):
        self.set_mode(mode=0)
        for epoch_i in range(epochs):
            epoch_start_time = time.time()
            train_loss = self.one_epoch_encoder()
            print(
                '[%03d/%03d] %2.2f sec(s) Train Loss: %3.6f ' %
                (epoch_i + 1, epochs, (time.time() - epoch_start_time), train_loss))
        for epoch_i in range(epochs):
            break

    def one_epoch_encoder(self) -> float:
        for _, data in enumerate(self.encoder_training_dataloader, 0):
            cell, positive_cell, negative_cell, confidence = data
            train_loss = self.encoder_loss(
                cell, positive_cell, negative_cell, confidence)
            self.first_stage_optimizer.zero_grad()
            train_loss.backward()
            self.first_stage_optimizer.step()
        return train_loss

    def one_epoch_decoder(self, use_gpu: bool = False):
        # for _, data in enumerate(self.encoder_training_dataset):
        triplet_loss = nn.MarginRankingLoss(margin=0.5)
        mse_loss = nn.MSELoss()

    def encoder_loss(self, cell, positive_cell, negative_cell, confidence):
        zero = torch.Tensor([0.0]).to(self.use_device)
        cell, positive_cell, negative_cell, confidence = \
            cell.to(self.use_device),\
            positive_cell.to(self.use_device),\
            negative_cell.to(self.use_device), \
            confidence.to(self.use_device)
        self.encoder_triplet_network = \
            self.encoder_triplet_network.to(self.use_device)

        dist_positive, dist_negative, _, _, _ = self.encoder_triplet_network(
            cell, positive_cell, negative_cell)
        train_loss = torch.mean(
            torch.max(dist_positive - dist_negative + confidence, zero))
        return train_loss
