import pandas as pd

data = {'colA': [10, 20, 30], 'colB': [40, 50, 60]}
df = pd.DataFrame(data)

df['new_column_of_ones'] = 1
