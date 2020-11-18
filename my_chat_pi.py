#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:44:28 2019

@author: xubing
"""

from aip import AipSpeech


APP_ID = '17831363'
API_KEY = 'N53gvEWGYoGa4QqHKIi7wMYB'
SECRET_KEY = 'SKSmIDjoI1sPM0KQ3oYiKMOYXrB4Qm4j'
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

#语音识别
# 读取文件
filePath = 'auido_5.pcm'
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


# 识别本地文件
text = client.asr(get_file_content(filePath), 'pcm', 16000, {
    'dev_pid': 1537,
})
    
    

##语音合成

#result  = client.synthesis('我叫徐一, 今天刚上幼儿园, 我今年3岁了', 'zh', 4, {
#        'spd' : 3,#语速
#        'pit' : 7, #音调
#        'vol': 7, #音量
#})
#
## 识别正确返回语音二进制 错误则返回dict 参照下面错误码
#if not isinstance(result, dict):
#    with open('auido_5.wav', 'wb') as f:
#        f.write(result)