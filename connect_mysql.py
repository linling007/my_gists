import pymysql
import pandas as pd

conn = {'host': '192.168.20.5', 'port': 3306, 'user': 'root', 'password': 'welcome1', 'db': 'event_series'}
# 打开数据库连接
db = pymysql.connect(**conn)

# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()

# 使用 execute()  方法执行 SQL 查询
cursor.execute("SELECT * from TMP_RAW_STRUCTED_FEATURES_2018_06_29_MODELING")

# 使用 fetchone() 方法获取单条数据.
data = cursor.fetchall()
df = pd.DataFrame(data, columns=[it[0] for it in cursor.description])
df.rename(columns={col: col.lower() for col in df}, inplace=True)
df.to_csv('raw_data_2018_06_29_modeling_v2.csv')
# print("Database version : %s " % data)
print(df.head())
# 关闭数据库连接
db.close()