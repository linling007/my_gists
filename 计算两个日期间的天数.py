'''
version: V0.1
Author: xubing
Date: 2020-11-10 18:20:23
LastEditors: xubing
LastEditTime: 2020-11-10 18:28:15
Description: 计算两个字符串时间的天数间隔、月数间隔
'''
from datetime import datetime


def calc_days(date1, date2=datetime.strftime(datetime.now(), '%Y-%m-%d')):
    date1 = datetime.strptime(date1[0:10], "%Y-%m-%d")
    date2 = datetime.strptime(date2[0:10], "%Y-%m-%d")
    num = (date2 - date1).days
    return num


def calc_months(date1, date2=datetime.strftime(datetime.now(), '%Y-%m-%d')):
    year1 = datetime.strptime(date1[0:10], "%Y-%m-%d").year
    year2 = datetime.strptime(date2[0:10], "%Y-%m-%d").year
    month1 = datetime.strptime(date1[0:10], "%Y-%m-%d").month
    month2 = datetime.strptime(date2[0:10], "%Y-%m-%d").month
    num = (year2 - year1) * 12 + (month2 - month1)
    return num


print(calc_months('2019-01-01'))