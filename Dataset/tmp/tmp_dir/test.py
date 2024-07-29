import pandas as pd
import numpy as np
import ast
## read the data
data = pd.read_csv('/home/yashjonjale/Documents/Dataset/tmp_dir/test.csv')
print(data.head())


## extract columns as lists
num = data['num_value_am'].tolist()
# print(num)
data['lst'] = data['lst'].apply(ast.literal_eval)
lst = data['lst'].tolist()
print(lst)