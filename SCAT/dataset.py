import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .triple import get_encoder_anchor_list, get_decoder_anchor_list


class EncoderTrainingDataset(Dataset):
    def __init__(
            self,
            data: pd.Series,
            metadata: pd.Series,
            resize_factor: float = 1000.0, anchor_score: float = 1.0):
        gene_num = len(data)
        cell_num = len(metadata)

        count_data = []
        for cell in range(cell_num):
            count_data.append(data[metadata['cell'][cell]])
        count_data = np.array(count_data, dtype=np.float32)
        count_data_row_sum = count_data.sum(axis=1)[:, None]
        normalized_data = count_data / count_data_row_sum * resize_factor

        positive_anchor, negative_anchor = get_encoder_anchor_list(metadata)

        self.data = normalized_data
        self.metadata = metadata
        self.gene_num = gene_num
        self.cell_num = cell_num
        self.positive_anchor = positive_anchor
        self.negative_anchor = negative_anchor
        self.anchor_score = anchor_score

    def __getitem__(self, item):
        item_batch = self.metadata["batch"][item]
        item_type = self.metadata["type"][item]
        positive_anchor = self.positive_anchor[(item_batch, item_type)][0]
        negative_anchor = self.negative_anchor[(item_batch, item_type)][0]
        positive_num = len(positive_anchor)
        negative_num = len(negative_anchor)
        if positive_num > 0:
            positive_item = positive_anchor[np.random.randint(positive_num)]
            positive_quality = self.anchor_score
        else:
            positive_item = item
            positive_quality = 0.0
        if negative_num > 0:
            negative_item = negative_anchor[np.random.randint(negative_num)]
            negative_quality = self.anchor_score
        else:
            negative_item = item
            negative_quality = 0.0

        cell = self.data[item]
        positive_cell = self.data[positive_item]
        negative_cell = self.data[negative_item]
        anchor_quality = torch.tensor(
            [(positive_quality + negative_quality) / 2], dtype=torch.float32)

        return cell, positive_cell, negative_cell, anchor_quality

    def __len__(self):
        return self.cell_num


class DecoderTrainingDataset(Dataset):
    def __init__(self, dataset: EncoderTrainingDataset):

        positive_anchor, negative_anchor = get_decoder_anchor_list(
            dataset.metadata)

        self.data = dataset.data
        self.metadata = dataset.metadata
        self.gene_num = dataset.gene_num
        self.cell_num = dataset.cell_num
        self.positive_anchor = positive_anchor
        self.negative_anchor = negative_anchor

    def __getitem__(self, item):
        item_batch = self.metadata["batch"][item]
        item_cluster = self.metadata["cluster"][item]
        positive_anchor = self.positive_anchor[(item_batch, item_cluster)][0]
        negative_anchor = self.negative_anchor[(item_batch, item_cluster)][0]
        positive_num = len(positive_anchor)
        negative_num = len(negative_anchor)
        positive_item = positive_anchor[np.random.randint(positive_num)] \
            if positive_num > 0 else item
        negative_item = negative_anchor[np.random.randint(negative_num)] \
            if negative_num > 0 else item

        cell = self.data[item]
        positive_cell = self.data[positive_item]
        negative_cell = self.data[negative_item]

        return cell, positive_cell, negative_cell, item, positive_item, negative_item

    def __len__(self):
        return self.cell_num


class TestingDataset(Dataset):
    def __init__(
            self,
            data: pd.Series,
            metadata: pd.Series,
            resize_factor: float = 1000.0):
        gene_num = len(data)
        cell_num = len(metadata)

        count_data = []
        for cell in range(cell_num):
            count_data.append(data[metadata['cell'][cell]])
        count_data = np.array(count_data, dtype=np.float32)
        count_data_row_sum = count_data.sum(axis=1)[:, None]
        normalized_data = count_data / count_data_row_sum * resize_factor

        self.data = normalized_data
        self.metadata = metadata[["batch", "type"]]
        self.gene_num = gene_num
        self.cell_num = cell_num

    def __getitem__(self, item):
        cell = self.data[item]
        return cell, item

    def __len__(self):
        return self.cell_num
