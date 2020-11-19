'''
version: V0.1
Author: xubing
Date: 2020-11-17 17:22:30
LastEditors: xubing
LastEditTime: 2020-11-18 11:57:37
Description: 
'''

from pyhive import hive


def dhive():
    try:
        conn = hive.connect(host="server_ip",
                            port=10000,
                            auth="...",
                            database="...",
                            username="...",
                            password="...")
        cursor = conn.cursor()
        cursor.execute("select * from table_name")
        res = cursor.fetchall()
        conn.close()
        for item in res:
            print(item)
    except Exception:
        print('excepion happen')


if __name__ == "__main__":
    dhive()