My Drive
Today
5:36 PM
O
You uploaded an item
Text
kmeans.py
Last week
Thu 9:53 PM
O
You edited an item
Google Sheets
SENIOR PROJECT IDEAS
Thu 10:13 AM
O
You edited an item
Google Docs
Management Paper 2
Wed 5:36 PM
O
You edited an item
Google Docs
Management Paper 2
Wed 3:28 PM
O
You edited an item
Google Docs
Management Paper 1
Wed 3:12 PM
O
You created an item
Google Docs
Management Paper 2
Tue 4:49 PM
O
You edited an item
Google Sheets
SENIOR PROJECT IDEAS
Earlier this month
Sep 13

Dan N edited an item
Google Sheets
SENIOR PROJECT IDEAS
Sep 13
O
You uploaded an item
Compressed Archive
HW1.zip
Sep 13
O
You created an item
Google Docs
Untitled document
Sep 11
O
You edited an item
Google Docs
Augmented Reality Billiards Proposal
Sep 11
O
You edited an item
Google Slides
Augmented Reality Billiards
Sep 11
O
You renamed an item
Google Slides
Augmented Reality Billiards
Untitled presentation
Sep 11
O
You edited an item
Google Sheets
SENIOR PROJECT IDEAS
Sep 10
O
You edited an item
Google Slides
Untitled presentation
Sep 10
O
You edited an item
Google Docs
Augmented Reality Billiards Proposal
Sep 10
O
You created an item
Google Slides
Untitled presentation
Sep 10
O
You edited an item
Google Docs
Augmented Reality Billiards Proposal
Sep 9
O
You and Dan N edited an item
Google Sheets
SENIOR PROJECT IDEAS
Sep 8
O
You and Dan N edited an item
Google Sheets
SENIOR PROJECT IDEAS
import train
import math
import pprint
import os
import pathlib
import numpy as np

l = []
i=1
k = 10

#take in file name of corp 2 or 3 
print('Please print Test Data List for corp 2 or 3')
test = input()

tempDir = './nlpTemp'

pathlib.Path(tempDir).mkdir(parents=True,exist_ok=True)

with open(test) as f:
    for line in f:
        l.append(line.split(" "))
        i+=1
f.close()

np.random.shuffle(l)

size = math.floor(i/k)
split = []
temp = []

for j in range (0,k):
    for m in range(0,size):
        temp.append(l[m*k+j])
    split.append(temp)
    temp = []

trainTestCompare={}

trainTestCompare['train']=[]
trainTestCompare['test']=[]
trainTestCompare['compare']=[]
cmpr =[]
cmprt = []
# this will create files to check
for i in range(0,k): 
    trainTestCompare['test'].append('./nlpTemp/test'+str(i))
    o = open('./nlpTemp/test'+str(i),'w')  
    for j in range (0,size):
        o.write(split[i][j][0]+'\n')
    o.close()
for i in range(0,k):
    trainTestCompare['compare'].append('./nlpTemp/compare'+str(i))
    o = open('./nlpTemp/compare'+str(i),'w')  
    cmprt = []
    for j in range (0,size):
        o.write(split[i][j][0]+' '+split[i][j][1])
        cmprt.append([split[i][j][0],split[i][j][1]])# this will be 10 "files" each file has list of
        #pairs and each pair is file then value 
    cmpr.append(cmprt)
    o.close()
for i in range(0,k):
    trainTestCompare['train'].append('./nlpTemp/train'+str(i))
    o = open('./nlpTemp/train'+str(i),'w')  
    for p in range(0,k):
        if p != i:
            for j in range (0,size):
                o.write(split[p][j][0]+' '+split[p][j][1])
    o.close()


pp = pprint.PrettyPrinter(indent = 4) #setup pprint

# pp.pprint(trainTestCompare)
out = []
for i in range(0,k):
    out.append('out'+str(i))

def comp(real,generated):
    correct=[]
    wrong = []
    total = []

    print(len(real))
    print(len(generated))

    for i in range(0,len(generated)):#should be deciding what test
        wrong.append(0)
        correct.append(0)
        total.append(0)
        for j in range(0,len(generated[i])): #should decide what q
            if(generated[i][j][1]==real[i][j][1]):
                correct[i] +=1
            else:
                wrong[i]+=1
            total[i]+=1
    return total, correct 

f=[]
for i in range(0,k):
    print ('training and testing set '+ str(i))
    result,_,_,_= train.TrainandTest(trainTestCompare['train'][i],trainTestCompare['test'][i],out[i])
    f.append(result)
print(len(f))
t, c = comp(cmpr,f)
percent = []

for i in range(0,len(t)):
    percent.append((c[i]/t[i]))

pp.pprint(percent)
done = 0
for i in range(0,len(t)):
    done += percent[i]

done /= len(t)

pp.pprint(done)