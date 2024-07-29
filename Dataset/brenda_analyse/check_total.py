import pandas as pd
import numpy as np
import ast
import os



path = os.getcwd()+"/total.csv"

df = pd.read_csv(path)

# get the unique seq_str
seq_str = df['seq_str'].unique()
#number of unique tuples of seq_str and ec_id combination
st = set()
for index, row in df.iterrows():
    st.add((row['seq_str'], row['substrate'], row['num_value_gm'], row['EC_ID']))
print(f"Total unique seq_str and substrate: {len(st)}")
print(f"Total unique seq_str: {len(seq_str)}")
print(f"Total rows: {len(df)}")