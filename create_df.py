import os
import pandas as pd
import pickle

folder = 'train_preprocessed'
files = sorted([fname for fname in os.listdir('./train_preprocessed') if 'data' in fname])

df = pd.DataFrame()
for fname in files:
    data = pd.read_csv(os.path.join(folder,fname), header=0, index_col=[0,1,2])
    events = pd.read_csv(os.path.join(folder,fname.replace('data','events')), header=0, index_col=[0,1,2])
    merged = pd.concat([data,events],axis=1)
    df = df.append(merged)

pickle.dump(df, open('eeg_raw.p', 'wb'))
