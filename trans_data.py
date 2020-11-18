#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 18:20:43 2019

@author: xubing
"""

import pandas as pd
file = 'DATA_0724.csv'
df_s = pd.read_csv(file,sep=';')
df_s.to_csv('DATA_0724_com.csv',index=False,encoding='utf8')