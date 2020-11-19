import pandas as pd
divide_line = '*' * 20
raw_file = 'raw_data_2018_06_29_v1.csv'
raw_df = pd.read_csv(raw_file)
df = raw_df[raw_df.columns[4:]]
df.drop_duplicates(subset='NAME', inplace=True)
print(divide_line)
print('数据shape：', df.shape)
print(divide_line)
# print(df['ISBAD'].value_counts())
print(df.head())