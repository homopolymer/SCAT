import numpy as np
import SCAT
import pandas as pd
from SCAT.model import evaluation

data = pd.read_csv('data.csv')
metadata = pd.read_csv('metadata.csv')
data.rename(columns={'Unnamed: 0': 'gene'}, inplace=True)
metadata.rename(columns={'Unnamed: 0': 'cell'}, inplace=True)

scat = SCAT.SCAT(data=data, metadata=metadata, num_workers=8)
scat.train()

test_data = data  # input your own data
test_metadata = metadata

output = evaluation(
    scat,
    data=test_data,
    metadata=test_metadata,
    num_workers=8,
    use_gpu=True)

np.savetxt('output.csv', output)
