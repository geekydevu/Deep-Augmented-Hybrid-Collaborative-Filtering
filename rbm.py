import numpy as np
import math
from random import shuffle
import pandas as pd
import copy

def logistic(x) :
    return 1.0/(1 + np.exp(-x))

class RBM :

    def __init__(self, numdims, lw=0.001, lb=0.001, lh=0.001, wc=0.01, initialmomentum=0.5, finalmomentum=0.9, maxepoch=5, numhids=100, k=5):

        self.numhids = numhids
        self.epsilonw = lw   # learning rate for weights
        self.epsilonvb = lb  # learning rate for biases for visible units
        self.epsilonhb = lh  # learning rate for biases for hidden units
        self.weightcost = wc
        self.initialmomentum = initialmomentum
        self.finalmomentum = finalmomentum
        self.maxepoch = maxepoch
        self.k = k
        self.numdims = numdims
        self.momentum = 0.0

    def f(self, X) :
    	return np.sum(X*np.array([i for i in range(1,self.k+1)]))

    def sample_hidden(self, v) :
    	pos_hid_activations = np.zeros(self.numhids)
    	for en  in v :
    		for k in range(self.k) :
    			pos_hid_activations += self.Wijk[k][en[0]]*en[1][k]
    	pos_hid_activations += self.hidbiaises
        pos_hid_probs = np.apply_along_axis(logistic,0, pos_hid_activations)
        pos_hid_states = pos_hid_probs > 0.5
        return pos_hid_probs, pos_hid_states

    def sample_visible(self, h, v) :
    	v1 = copy.deepcopy(v)
    	for i in range(len(v)) :
    		for k in range(self.k) :
    			v1[i][1][k] = np.sum(self.Wijk[k][v1[i][0]] * h)
    		v1[i][1] += self.visbiaises[:,v1[i][0]]
    	for i in range(len(v)) :
    		mid = v1[i][0]
    		rat = v1[i][1]
    		rat = np.exp(rat)
    		sum1 = sum(rat)
    		rat /= sum1
    		v1[i][1] = rat
    	return v1

    def predictForward(self, X_train) :

		hid_states = np.zeros([self.num_train, self.numhids])
		for i in range(self.num_train) :
		    hid_states[i],tmp = self.sample_hidden(X_train[i])
		return hid_states

    def predictBackward(self, X_train, hid_states) :

		#print hid_states.shape
		negdata = copy.deepcopy(X_train)
		for i in range(self.num_train) :
			negdata[i] = self.sample_visible(hid_states[i],X_train[i])
		return negdata

    def fit(self, X_train,X_val) :

    	self.num_train = len(X_train)
    	self.Wijk = np.random.normal(size=(self.k, self.numdims, self.numhids))*(0.1)

        self.hidbiaises = np.zeros(self.numhids)
        self.visbiaises = np.zeros([self.k,self.numdims])

        Wijk_inc = np.zeros([self.k, self.numdims, self.numhids])
        hidbiaises_inc = np.zeros(self.numhids)
        visbiaises_inc = np.zeros([self.k,self.numdims])

        for epoch in range(1, self.maxepoch + 1) :

            err=0.0
            cnt=0
            err2=0.0
            cnt2=0.0
            momentum = 0.6
            if epoch > 9 : momentum = 0.9
            #print np.sum(self.Wijk), np.sum(self.visbiaises), np.sum(self.hidbiaises)
            print 'Starting epoch ', epoch

            for uid in range(self.num_train) :

            	pos_hid_probs, pos_hid_states = self.sample_hidden(X_train[uid])
            	negdata = self.sample_visible(pos_hid_states, X_train[uid])
            	neg_hid_probs, neg_hid_states = self.sample_hidden(negdata)
            	new_tmp = copy.deepcopy(negdata)
            	T=1
            	if epoch > 9 : T = 3
                if epoch > 15 : T = 5
            	for i in range(T-1) :
            		pos_hid_probs2, pos_hid_states2 = self.sample_hidden(new_tmp)
            		negdata = self.sample_visible(pos_hid_states2, new_tmp)
            		neg_hid_probs, neg_hid_states = self.sample_hidden(negdata)
            		new_tmp = copy.deepcopy(negdata)
                for i in range(len(negdata)) :
                    cnt+=1
                    posprod = np.outer(X_train[uid][i][1], pos_hid_probs)
                    negprod = np.outer(negdata[i][1], neg_hid_probs)

                    Wijk_inc[:,negdata[i][0],:] = momentum*Wijk_inc[:,negdata[i][0]] + self.epsilonw * (posprod-negprod) - self.Wijk[:,negdata[i][0],:]*self.weightcost*self.epsilonw
                    visbiaises_inc[:,negdata[i][0]] = momentum*visbiaises_inc[:,negdata[i][0]] + self.epsilonvb * (X_train[uid][i][1] - negdata[i][1]) - self.weightcost*self.epsilonvb*self.visbiaises[:,negdata[i][0]]
                    hidbiaises_inc = momentum*hidbiaises_inc + self.epsilonhb * (pos_hid_probs - neg_hid_probs) - self.weightcost*self.hidbiaises*self.epsilonhb
                    self.hidbiaises += hidbiaises_inc
                    self.Wijk[:,negdata[i][0],:] += Wijk_inc[:,negdata[i][0],:]
                    self.visbiaises[:,negdata[i][0]] += visbiaises_inc[:,negdata[i][0]]
                    error = abs(self.f(negdata[i][1]) - self.f(X_train[uid][i][1]))
                    err += error
                negdata2 = self.sample_visible(pos_hid_states, X_val[uid])
                for i in range(len(negdata2)) :
            	    cnt2+=1
            	    err2 += abs(self.f(negdata2[i][1]) - self.f(X_val[uid][i][1]))
            f_error = err/cnt
            print 'Epoch error : ', f_error
            print 'Epoch error : ', err2/cnt2
            self.epsilonw *= 0.95
            self.epsilonhb *= 0.95
            self.epsilonvb *= 0.95

        print 'Training complete .... '
        np.save('weights_50_125_200_items/l1_weights.npy',self.Wijk)
    	np.save('weights_50_125_200_items/l1_vis.npy',self.visbiaises)
    	np.save('weights_50_125_200_items/l1_hid.npy',self.hidbiaises)
