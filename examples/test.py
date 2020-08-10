import SCAT

import pandas as pd
filedir = '/home/zhangjl/datasets/TripletTest/B/SCAT_pancreas/1'
data = pd.read_csv(filedir + '/fdata.csv')
metadata = pd.read_csv(filedir + '/fmeta.csv')
#pair = pd.read_csv(filedir + '/fpair.csv').drop('Unnamed: 0', 1)
data.rename(columns={'Unnamed: 0': 'gene'}, inplace=True)
metadata.rename(columns={'Unnamed: 0': 'cell'}, inplace=True)
print(data.shape)

scat = SCAT.SCAT(data=data, metadata=metadata, num_workers=8)
scat.train()
