import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# Define globals and import dataset
mnist = tf.keras.datasets.mnist # lol MFW i tried to download mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train,x_test = x_train/255.0 ,x_test/255.0		# go from 0-1 instead of 0-255 8 bit greyscale 
y_train = y_train.astype('int32')					# fix due to tensorflow complaining
y_test = y_test.astype('int32') 					# fix due to tensorflow complaining



def plotVal():
	ln,_,_ = x_train.shape

	rn = np.random.randint(0,ln-1)
	#rn = 26563
	print(ln,rn)
	test_val=x_train[rn]

	fig1= plt.figure(1)
	drx=6 #pixels top bottom left and right to drop
	dry=4
	xc,yc = np.linspace(0+drx,27-drx,28-2*drx),np.linspace(27-dry,0+dry,28-2*dry) 
	xv,yv = np.meshgrid(xc,yc)

	print(y_train[rn], s[rn])
	
	#reduce dimentions of the test data
	fx,lx = 0+drx, 28-drx
	fy,ly = 0+dry, 28-dry


	z = test_val[fy:ly,fx:lx]

	CS = plt.contourf(xv,yv,z,cmap='gray')
	w= plt.xlabel('w')
	h= plt.ylabel('h')
	h.set_rotation(0)
	plt.title("MNIST")
	plt.axis('equal') 
	#plt.clabel(CS, fontsize=9, inline=1)
	plt.show()

z = x_train.shape[0]# 60000 hopefully
s = np.arange(z)
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]


def onehotBin(y):
	l = y.shape[0]
	print(l)
	yo = []
	for z in range(l):
		if (y[z] ==0):
			yo.append([0,0,0,0])
		elif (y[z] ==1):
			yo.append([0,0,0,1])
		elif (y[z] ==2):
			yo.append([0,0,1,0])
		elif (y[z] ==3):
			yo.append([0,0,1,1])
		elif (y[z] ==4):
			yo.append([0,1,0,0])
		elif (y[z] ==5):
			yo.append([0,1,0,1])
		elif (y[z] ==6):
			yo.append([0,1,1,0])
		elif (y[z] ==7):
			yo.append([0,1,1,1])
		elif (y[z] ==8):		
			yo.append([1,0,0,0])
		elif (y[z] ==9):
			yo.append([1,0,0,1])
			
	return yo


lx = x_train.shape[0]
nv =int( lx *.2)
nt = lx - nv

s = np.asarray(onehotBin(y_train))
print(s.shape)
plotVal()

#plotVal()