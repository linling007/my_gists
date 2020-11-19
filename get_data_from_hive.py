'''
version: V0.1
Author: xubing
Date: 2020-11-18 13:54:22
LastEditors: xubing
LastEditTime: 2020-11-18 14:08:11
Description: 
'''
from pyhive import hive
# from impala.dbapi import connect
conn = hive.connect(
    host="192.168.100.202",
    port=10000,
    database='dm_tag',
    auth=None,
)
cur = conn.cursor()
query_sql = "select * from dm_fintag_h00_cr limit 10"
cur.execute(query_sql)
data = cur.fetchall()
print(data)
cur.close()
conn.close()

# try:
#     conn = hive.connect(
#         host="192.168.100.202",
#         port=10000,
#     )
#     # auth="...",
#     # database="...",
#     # username="...",
#     # password="...")
#     cursor = conn.cursor()
#     cursor.execute("select * from dm_tag.dm_fintag_h00_cr limit 10")
#     res = cursor.fetchall()
#     conn.close()
#     for item in res:
#         print(item)
# except Exception:
#     print('excepion happen')
