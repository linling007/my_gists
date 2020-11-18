"""
Created by xubing on 2020/4/27

"""
import logging
import time
def use_logging(func):

    def wrapper():
        # logging.error("%s is running" % func.__name__)
        stime = time.time()
        func()
        etime = time.time()
        print(etime - stime)

        return func()
    return wrapper

@use_logging
def foo():
    time.sleep(3)
    print("i am foo")

foo()