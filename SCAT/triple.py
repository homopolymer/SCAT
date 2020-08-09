import pandas as pd
import numpy as np
from collections import defaultdict


def get_anchor_list(metadata: pd.Series):
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
        positive_anchor[(batch, cell_type)].append(
            np.intersect1d(batch_mismatch, type_match))
        negative_anchor[(batch, cell_type)].append(type_mismatch)
    return positive_anchor, negative_anchor


filedir = '/home/zhangjl/datasets/TripletTest/B/SCAT_simu/1'
meta = pd.read_csv(filedir + '/fmeta.csv')
a, b = get_anchor_list(meta)
