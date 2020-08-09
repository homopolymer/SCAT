import torch.nn as nn


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
    def __init__(self, gene_num, latent_size=10, dropout_rate=0.0):
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
    def __init__(self, gene_num: int, latent_size: int = 10):
        super().__init__()
        self.decoder_r = nn.Sequential(
            nn.Linear(latent_size, gene_num),
            nn.ReLU(inplace=True)
        )

    def forward(self, network_input):
        output = self.decoder_r(network_input)
        return output


class DecoderD(nn.Module):
    def __init__(self, gene_num: int, one_hot_num: int, latent_size: int = 10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(latent_size + one_hot_num, 100),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1000, gene_num)
        )

    def forward(self, network_input):
        output = self.layer3(self.layer2(self.layer1(network_input)))
        return output
