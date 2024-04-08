import pickle as pkl
import pandas as pd

with open("dataset.pkl", "rb") as f:
    obj = pkl.load(f)

df = pd.DataFrame(obj)
df.to_csv(r'file.csv')
