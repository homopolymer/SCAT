import numpy as np
import pandas as pd
from SCAT import SCAT, evaluation, set_seed

data = pd.read_csv('data.csv')
metadata = pd.read_csv('metadata.csv')
data.rename(columns={'Unnamed: 0': 'gene'}, inplace=True)
metadata.rename(columns={'Unnamed: 0': 'cell'}, inplace=True)

scat = SCAT(
    data=data,
    metadata=metadata,
    num_workers=4,
    use_gpu=True,
    dropout_rate=0.0)
scat.train(epochs=50)

test_data = data  # input your own data
test_metadata = metadata

output = evaluation(
    scat,
    data=test_data,
    metadata=test_metadata,
    num_workers=4,
    use_gpu=True)

np.savetxt('output.csv', output)
