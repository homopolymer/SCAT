import pandas as pd
import numpy as np
from collections import defaultdict


def get_encoder_anchor_list(metadata: pd.Series):
    annotation = metadata[['batch', "type"]]
    annotation_group = annotation.groupby(["batch", "type"]).groups
    annotation_group_keys = list(annotation_group.keys())

    positive_anchor = defaultdict(list)
    negative_anchor = defaultdict(list)

    for batch, cell_type in annotation_group_keys:
        batch_mismatch = np.array(
            annotation[annotation['batch'] != batch].index, dtype=np.int64)
        type_match = np.array(
            annotation[annotation['type'] == cell_type].index, dtype=np.int64)
        type_mismatch = np.array(
            annotation[annotation['type'] != cell_type].index, dtype=np.int64)
        positive_match = np.intersect1d(batch_mismatch, type_match)
        if len(positive_match) > 0:
            positive_anchor[(batch, cell_type)].append(positive_match)
        else:
            positive_anchor[(batch, cell_type)].append(type_match)
        negative_anchor[(batch, cell_type)].append(type_mismatch)
    return positive_anchor, negative_anchor


def get_decoder_anchor_list(metadata: pd.Series):
    annotation = metadata[['batch', "cluster"]]
    annotation_group = annotation.groupby(["batch", "cluster"]).groups
    annotation_group_keys = list(annotation_group.keys())

    positive_anchor = defaultdict(list)
    negative_anchor = defaultdict(list)

    for batch, cluster in annotation_group_keys:
        batch_mismatch = np.array(
            annotation[annotation["batch"] != batch].index, dtype=np.int64)
        batch_match = np.array(
            annotation[annotation["batch"] == batch].index, dtype=np.int64)
        cluster_match = np.array(
            annotation[annotation["cluster"] == cluster].index, dtype=np.int64)
        positive_anchor[(batch, cluster)].append(
            np.intersect1d(batch_match, cluster_match))
        negative_anchor[(batch, cluster)].append(batch_mismatch)
    return positive_anchor, negative_anchor
