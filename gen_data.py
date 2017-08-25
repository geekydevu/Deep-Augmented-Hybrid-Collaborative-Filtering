import numpy as np 
import pandas as pd 

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


users = np.load('users_75_rbm.npy')
items = np.load('items_50_rbm.npy')

X_train = np.zeros([len(transactions), 125])
Y_train = np.zeros(len(transactions))

for i in range(len(transactions)) : 
	if i%100000==0 : 
		print i
	uid = u2h[transactions['userId'][i]]
	iid = i2h[transactions['itemId'][i]]
	rating = transactions['rating'][i]-1
	X_train[i] = np.concatenate((users[uid], items[iid]))
	Y_train[i] = rating

X_test = np.zeros([len(val_transactions), 125])
Y_test = np.zeros(len(val_transactions))

for i in range(len(val_transactions)) : 
	if i%100000==0 : 
		print i
	uid = u2h[val_transactions['userId'][i]]
	iid = i2h[val_transactions['itemId'][i]]
	rating = val_transactions['rating'][i]-1
	X_test[i] = np.concatenate((users[uid], items[iid]))
	Y_test[i] = rating

np.save('metadata/x_train_probs.npy', X_train)
np.save('metadata/x_test_probs.npy', X_test)
np.save('metadata/y_train_probs.npy', Y_train)
np.save('metadata/y_test_probs.npy', Y_test)