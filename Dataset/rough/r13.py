import pandas as pd
import numpy as np

# # Sample data
# data = {
#     'ID': [1, 2, 3],
#     'Vector': 
# }

# Create a DataFrame
# df = pd.DataFrame(data)

# # Display the DataFrame
# print("Original DataFrame:")
# print(df)
# print(df.head())

# # Convert NumPy arrays to a string representation
# df['Vector'] = df['Vector'].apply(lambda x: np.array2string(x, separator=','))

# # Display the modified DataFrame
# print("\nDataFrame with NumPy arrays stored as strings:")
# print(df)

# # Convert the string representation back to NumPy arrays
# df['Vector'] = df['Vector'].apply(lambda x: np.fromstring(x[1:-1], sep=','))

# # Display the final DataFrame
# print("\nFinal DataFrame with NumPy arrays:")
# print(df)


lst = [np.random.rand(5), np.random.rand(5), np.random.rand(5)]
# print(lst)

for x in lst:
    print(x)
    print(np.array2string(x, separator=','))
    print(np.fromstring(np.array2string(x, separator=','), sep=','))
    input()