import numpy as np
import math
import pandas as pd
from rbm import *
from rbm1 import *
from random import shuffle

def f(X) :
	#print X.shape
	return np.sum(X*np.array([1,2,3,4,5]))

class DBN :

    def __init__(self, X, X2) :

		#positive phase
		rbm0 = RBM(maxepoch=25, numdims = len(items), numhids=75)
		rbm0.fit(X,X2)
		H1 = rbm0.predictForward(X)
		np.save('users_75_rbm.py', H1)

		# rbm1 = RBM1(numdims = 50, numhids=125, maxepoch=15,l=2,rbms=[rbm0])
		# rbm1.fit(H1,X2)
		# H2 = rbm1.predictForward(H1)

		# rbm2 = RBM1(numdims=125,numhids=200, maxepoch=15,l=3,rbms=[rbm0,rbm1])
		# rbm2.fit(H2,X2)
		# H3 = rbm2.predictForward(H2)
		# np.save('items_200_dbn.py', H3)

		# #negative phase
		# H2_dash = rbm2.predictBackward(H3)
		# H1_dash = rbm1.predictBackward(H2_dash)
		X_dash = rbm0.predictBackward(X,H1)


		X2_dash = rbm0.predictBackward(X2,H1)

		err=0.0
		cnt=0
		for i in range(len(X)) :
			for j in range(len(X[i])) :
				cnt+=1
				err += abs(f(X_dash[i][j][1]) - f(X[i][j][1]))

		print 'Error : ', err/cnt

		err=0.0
		cnt=0
		for i in range(len(X2)) :
			for j in range(len(X2[i])) :
				cnt += 1
				err += (f(X2_dash[i][j][1]) - f(X2[i][j][1]))**2

		print 'Error :', math.sqrt(err/cnt)


if __name__ == '__main__':


	#Read tables
	users = pd.read_csv('datasets/ml-100k/data2/user_data.csv')
	items = pd.read_csv('datasets/ml-100k/data2/item_data.csv')
	transactions = pd.read_csv('datasets/ml-100k/data2/train_ratings.csv')
	val_transactions = pd.read_csv('datasets/ml-100k/data2/test_ratings.csv')

	u2h = {}
	i=0
	for uid in users['userId'] :
	    u2h[uid]=i
	    i+=1

	i2h = {}
	i=0
	for iid in items['itemId'] :
	    i2h[iid]=i
	    i+=1

	h2u = {}
	i=0
	for uid in users['userId'] :
	    h2u[i]=uid
	    i+=1

	h2i = {}
	i=0
	for iid in items['itemId'] :
	    h2i[i]=iid
	    i+=1


	data = [[] for i in range(len(users))]
	for i in range(len(transactions)) :
		if i%100000==0 :
			print i
		uid = u2h[transactions['userId'][i]]
		iid = i2h[transactions['itemId'][i]]
		rating = transactions['rating'][i]-1
		tmp = np.zeros(5)
		tmp[rating]=1
		data[uid].append([iid, tmp])
	val_data = [[] for i in range(len(users))]
	for i in range(len(val_transactions)) :
		if i%100000==0 :
			print i
		uid = u2h[val_transactions['userId'][i]]
		iid = i2h[val_transactions['itemId'][i]]
		rating = val_transactions['rating'][i]-1
		tmp = np.zeros(5)
		tmp[rating]=1
		val_data[uid].append([iid,tmp])

	dbn = DBN(data,val_data)
