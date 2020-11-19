import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
# tips = sns.load_dataset("tips")
df = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'x2': [20, 30, 40, 50, 60, 10, 70, 80, 90, 100],
    'y': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
})
# ax = sns.boxplot(x=df['x'], orient='h')
for col in df:
    if col != 'y':
        df.boxplot(column=[col], by=['y'])
plt.show()