#!/bin/python3.5
# Ostap Voynarovskiy
# CGML HW4
# October 4 2018
# Professo Curro
import keras
import numpy as np
import pandas as pd

# Read Data
train = pd.read_csv("./ag_news/train.csv")
train.columns = ['class','title','description']
test = pd.read_csv("./ag_news/test.csv")
test.columns = ['class','title','description']
# generate word embedings 
train['space']=" "

train["text"] = train.title.map(str)+train.space+train.description.map(str)
print(train.text[2])

