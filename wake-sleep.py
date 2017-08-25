import numpy as np
import math
from random import shuffle
import copy
import pandas as pd
from numpy import random

def logistic(x) :
    return 1.0/(1 + np.exp(-x))

def f(X) :
    return np.sum(X*np.array([i for i in range(1,6)]))

def cal_wakehidprobs(v,vishid,hidrecbiases) :

	pos_hid_activations = np.zeros(50)
	for en  in v :
		for k in range(5) :
			pos_hid_activations += vishid[k][en[0]]*en[1][k]
	pos_hid_activations += hidrecbiases
	pos_hid_probs = np.apply_along_axis(logistic,0, pos_hid_activations)
	return pos_hid_probs

def cal_sleepvisprobs(v,h,hidvis,visgenbiases) :

	v1 = copy.deepcopy(v)
	for i in range(len(v)) :
		for k in range(5) :
			v1[i][1][k] = np.sum(hidvis[k,:,v1[i][0]] * h)
		v1[i][1] += visgenbiases[:,v1[i][0]]
	for i in range(len(v)) :
		mid = v1[i][0]
		rat = v1[i][1]
		rat = np.exp(rat)
		sum1 = sum(rat)
		rat /= sum1
		v1[i][1] = rat
	return v1

def finetune_weights(n_epochs, data, val_data) :

	epsilonw = 0.001
	epsilonvb = 0.001
	epshilonb = 0.001
	epshilonw = 0.001
	weightcost = 0.05
	weightcost2 = 0.05

	#load weights
	vishid = np.load('weights_50_125_200_items/l1_weights.npy')
	hidvis = np.zeros([5,vishid.shape[2],vishid.shape[1]])
	hidpen = np.load('weights_50_125_200_items/l2_weights.npy')
	penhid = copy.deepcopy(hidpen.T)
	pentop = np.load('weights_50_125_200_items/l3_weights.npy')
	hidrecbiases = np.load('weights_50_125_200_items/l1_hid.npy')
	penrecbiases = np.load('weights_50_125_200_items/l2_hid.npy')
	topbiases = np.load('weights_50_125_200_items/l3_hid.npy')
	pengenbiases = np.load('weights_50_125_200_items/l3_vis.npy')
	hidgenbiases = np.load('weights_50_125_200_items/l2_vis.npy')
	visgenbiases = np.load('weights_50_125_200_items/l1_vis.npy')

	vishid_inc = np.zeros_like(vishid)
	hidvis_inc = np.zeros_like(hidvis)
	hidpen_inc = np.zeros_like(hidpen)
	penhid_inc = np.zeros_like(penhid)
	pentop_inc = np.zeros_like(pentop)
	hidrecbiases_inc = np.zeros_like(hidrecbiases)
	penrecbiases_inc = np.zeros_like(penrecbiases)
	topbiases_inc = np.zeros_like(topbiases)
	pengenbiases_inc = np.zeros_like(pengenbiases)
	hidgenbiases_inc = np.zeros_like(hidgenbiases)
	visgenbiases_inc = np.zeros_like(visgenbiases)

	for i in range(5) :
		hidvis[i] = vishid[i].T
	#start contrastive wake-sleep algorithm
	for epoch in range(n_epochs) :
		momentum = 0.0
		T = 1
		print 'Starting epoch ', epoch + 1
		mae=0.0
		cnt=0
		mae2=0.0
		cnt2=0
		for uid in range(len(data)) :

			wakehidprobs = cal_wakehidprobs(data[uid],vishid,hidrecbiases)
			wakehidstates = wakehidprobs > 0.5
			#wakehidstates = random.binomial(size=wakehidprobs.shape, n=1, p=wakehidprobs)
			#wakepenprobs = np.dot(wakehidstates, hidpen) + penrecbiases
			wakepenprobs = np.apply_along_axis(logistic, 0, np.dot(wakehidstates, hidpen) + penrecbiases)
			wakepenstates = wakepenprobs > 0.5
			#wakepenstates = random.binomial(size=wakepenprobs.shape, n=1, p=wakepenprobs)

			#waketopprobs = np.dot(wakepenstates, pentop) + topbiases
			waketopprobs = np.apply_along_axis(logistic, 0, np.dot(wakepenstates, pentop) + topbiases)
			waketopstates = waketopprobs > 0.5
			#waketopstates = random.binomial(size=waketopprobs.shape, n=1, p=waketopprobs)

			# perform Gibbs sampling iterations using top-level undirected associative memory
			negtopstates = waketopstates
			for iter in range(T) :
				#negpenprobs = np.dot(negtopstates, pentop.T) + pengenbiases
				negpenprobs = np.apply_along_axis(logistic, 0, np.dot(negtopstates, pentop.T) + pengenbiases)
				negpenstates = negpenprobs > 0.5
				#negpenstates = random.binomial(size=negpenprobs.shape, n=1, p=negpenprobs)

				#negtopprobs = np.dot(negpenstates, pentop) + topbiases
				negtopprobs = np.apply_along_axis(logistic, 0, np.dot(negpenstates, pentop) + topbiases)
				negtopstates = negtopprobs > 0.5
				#negtopstates = random.binomial(size=negtopprobs.shape, n=1, p=negtopprobs)


			# perform top-down generative pass
			sleeppenstates = negpenstates

			#sleephidprobs = np.dot(sleeppenstates, penhid) + hidgenbiases
			sleephidprobs = np.apply_along_axis(logistic, 0, np.dot(sleeppenstates, penhid) + hidgenbiases)
			sleephidstates = sleephidprobs > 0.5
			#sleephidstates = random.binomial(size=sleephidprobs.shape, n=1, p=sleephidprobs)

			sleepvisprobs = cal_sleepvisprobs(data[uid],sleephidstates,hidvis,visgenbiases)

			# predictions
			#psleeppenstates = np.dot(sleephidstates, hidpen) + penrecbiases
			psleeppenstates = np.apply_along_axis(logistic, 0, np.dot(sleephidstates, hidpen) + penrecbiases)

			psleephidstates = cal_wakehidprobs(sleepvisprobs,vishid,hidrecbiases)
			pvisprobs = cal_sleepvisprobs(data[uid],wakehidstates,hidvis,visgenbiases)
			pvisprobs2 = cal_sleepvisprobs(val_data[uid],wakehidstates,hidvis,visgenbiases)
			#phidprobs = np.dot(wakepenstates, penhid) + hidgenbiases
			phidprobs = np.apply_along_axis(logistic, 0, np.dot(wakepenstates, penhid) + hidgenbiases)

			# updates to generative parameters
			for i in range(len(data[uid])) :
				tmp = abs(f(data[uid][i][1]) - f(pvisprobs[i][1]))
				mae2 += tmp
				cnt2 += 1
				#compute the fucking gradient
				hidvis_inc[:,:,data[uid][i][0]] = momentum * hidvis_inc[:,:,data[uid][i][0]] + epsilonw * np.outer((data[uid][i][1]-pvisprobs[i][1]), wakehidstates) - weightcost*epshilonw*hidvis[:,:,data[uid][i][0]]
				hidvis[:,:,data[uid][i][0]] += hidvis_inc[:,:,data[uid][i][0]]

				visgenbiases_inc[:,data[uid][i][0]] = momentum * visgenbiases_inc[:,data[uid][i][0]] + epsilonvb * (data[uid][i][1] - pvisprobs[i][1]) - weightcost2*epsilonvb*visgenbiases[:,data[uid][i][0]]
				visgenbiases[:,data[uid][i][0]] += visgenbiases_inc[:,data[uid][i][0]]
			for i in range(len(val_data[uid])) :
				tmp = abs(f(val_data[uid][i][1]) - f(pvisprobs2[i][1]))
				mae += tmp
				cnt += 1

			penhid_inc = momentum*penhid_inc + epshilonw * np.outer(wakepenstates, (wakehidstates-phidprobs)) - weightcost*epshilonw*penhid
			penhid += penhid_inc

			hidgenbiases_inc = momentum * hidgenbiases_inc + epshilonb * (wakehidstates - phidprobs) - weightcost2*epshilonb*hidgenbiases
			hidgenbiases += hidgenbiases_inc

			#updates to top-level associative parameters
			pentop_inc = momentum*pentop_inc + epshilonw * (np.outer(wakepenstates, waketopstates) - np.outer(negpenstates, negtopstates)) - weightcost*epshilonw*pentop
			pentop += pentop_inc

			pengenbiases_inc = momentum*pengenbiases_inc + epshilonb * (wakepenstates - negpenstates) - weightcost2*epsilonvb*pengenbiases
			pengenbiases += pengenbiases_inc

			topbiases_inc = momentum*topbiases_inc + epshilonb * (waketopstates - negtopstates) - weightcost2*epsilonvb*topbiases
			topbiases += topbiases_inc

			# updates to recoginition parameters
			for i in range(len(data[uid])) :
				#compute the fucking gradient
				vishid_inc[:,data[uid][i][0],:] = momentum*vishid_inc[:,data[uid][i][0],:] + epsilonw * np.outer(sleepvisprobs[i][1], (sleephidstates - psleephidstates)) - weightcost*epsilonw*vishid[:,data[uid][i][0],:]
				vishid[:,data[uid][i][0],:] += vishid_inc[:,data[uid][i][0],:]


			hidpen_inc = momentum*hidpen_inc + epshilonw * np.outer(sleephidstates, (sleeppenstates - psleeppenstates)) - weightcost*epsilonw*hidpen
			hidpen += hidpen_inc

			penrecbiases_inc = momentum*penrecbiases_inc + epshilonb * (sleeppenstates - psleeppenstates) - weightcost2*epshilonb*penrecbiases
			penrecbiases += penrecbiases_inc

			hidrecbiases_inc = momentum*hidrecbiases_inc + epshilonb * (sleephidstates - psleephidstates) - weightcost2*epshilonb*hidrecbiases
			hidrecbiases += hidrecbiases_inc

		print 'MAE :', mae/cnt
		print 'MAE(TRAINING) :', mae2/cnt2
		#epsilonw *= 0.95
		#epsilonvb *= 0.95
		#epshilonb *= 0.95
		#epshilonw *= 0.95


if __name__ == '__main__':


	#Read tables
	users = pd.read_csv('datasets/ml-100k/data5/user_data.csv')
	items = pd.read_csv('datasets/ml-100k/data5/item_data.csv')
	transactions = pd.read_csv('datasets/ml-100k/data5/train_ratings.csv')
	val_transactions = pd.read_csv('datasets/ml-100k/data5/test_ratings.csv')

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


	data = [[] for i in range(len(items))]
	for i in range(len(transactions)) :
		uid = u2h[transactions['userId'][i]]
		iid = i2h[transactions['itemId'][i]]
		rating = transactions['rating'][i]-1
		tmp = np.zeros(5)
		tmp[rating]=1
		data[iid].append([uid, tmp])
	val_data = [[] for i in range(len(items))]
	for i in range(len(val_transactions)) :
		if i%100000==0 :
			print i
		uid = u2h[val_transactions['userId'][i]]
		iid = i2h[val_transactions['itemId'][i]]
		rating = val_transactions['rating'][i]-1
		tmp = np.zeros(5)
		tmp[rating]=1
		val_data[iid].append([uid,tmp])

	#np.save('data.npy',data)
	#np.save('val_data.npy',val_data)
	#data = np.load('data.npy')
	#val_data = np.load('val_data.npy')

	finetune_weights(10, data, val_data)
