import pandas as pd
import numpy as np

data = {'colA': [10, 20, 30], 'colB': [40, 50, 60]}
df = pd.DataFrame(data)

df['new_column_of_ones'] = 1
z = df.values
n, k = z.shape #n is rows, k is columns, this finds the degrees of freedom (n-k)
#can't have more columns than rows
