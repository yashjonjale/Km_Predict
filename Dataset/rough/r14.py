import pandas as pd
import ast
import numpy as np

# Sample DataFrame
data = {
    'target_value': [1, 2, 3],
    'vars': ["[1, 2, 3]", "[4, 5, 6]", "[7, 8, 9]"]
}

df = pd.DataFrame(data)

# Convert the string representation of lists to actual lists
df['vars'] = df['vars'].apply(ast.literal_eval)

# Extract the 'target_value' column as a numpy array
target_values = df['target_value'].values

# Extract the 'vars' column, convert each list to a numpy array and stack them
vars_values = np.column_stack(df['vars'].values)

# Combine the 'target_value' and 'vars' columns into a single 2D array
result = np.column_stack((target_values, vars_values))

print(result)
