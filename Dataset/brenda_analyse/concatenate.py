import numpy as np
import pandas as pd
import glob
import os
import time

path = os.getcwd()+"/seqs/"

#make a common DataFrame
df_total = pd.DataFrame()

files = glob.glob(path + "/*.csv")
#sort files
files.sort()

print(f"Total files: {len(files)}")
i = 0
for file in files:
    i += 1
    if i % 100 == 0:
        print(f"Processing {i}th file") 
    df = pd.read_csv(file)
    for i in range(len(df)):
        if df.loc[i,'mutation'] == 'mutated':
            if df.loc[i,'change'] == '-':
                #delete the row in place and index reset
                df.drop(i, inplace=True)
                # df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df_total = pd.concat([df_total, df], ignore_index=True)
#delete the Unamed column
df_total = df_total.loc[:, ~df_total.columns.str.contains('^Unnamed')]
print(df_total.head())
df_total.to_csv(os.getcwd()+"/total.csv", index=False)