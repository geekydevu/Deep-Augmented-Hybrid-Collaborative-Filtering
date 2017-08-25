import numpy as np
import math
from random import shuffle
import pandas as pd
import copy

def logistic(x) :
    return 1.0/(1 + np.exp(-x))

def f(X) :
    return np.sum(X*np.array([i for i in range(1,5+1)]))

class RBM1 :

    def __init__(self, numdims, l, rbms=None, lw=0.001, lb=0.001, lh=0.001, wc=0.001, initialmomentum=0.5, finalmomentum=0.9, maxepoch=5, numhids=100):

        self.numhids = numhids
        self.epsilonw = lw   # learning rate for weights
        self.epsilonvb = lb  # learning rate for biases for visible units
        self.epsilonhb = lh  # learning rate for biases for hidden units
        self.weightcost = wc
        self.l=l
        self.initialmomentum = initialmomentum
        self.finalmomentum = finalmomentum
        self.maxepoch = maxepoch
        self.numdims = numdims
        self.momentum = 0.0
        self.rbms = rbms

    def fit(self, X_train, X_initial) :

        self.num_train = X_train.shape[0]
        self.numdims = X_train.shape[1]

        self.Wijk = np.random.normal(size=(self.numdims, self.numhids))*(0.1)
        self.hidbiaises = np.zeros(self.numhids)
        self.visbiaises = np.zeros(self.numdims)
        Wijk_inc = np.zeros([self.numdims, self.numhids])
        visbiais_inc = np.zeros(self.numdims)
        hidbiais_inc = np.zeros(self.numhids)
        negdata = np.zeros_like(X_train)
        negdata2 = np.zeros_like(X_train)

        for epoch in range(1,self.maxepoch+1) :
            momentum = 0.6
            #if epoch > 9 : momentum = 0.9

            err=0.0
            cnt=0
            #print np.sum(self.Wijk), np.sum(self.visbiaises), np.sum(self.hidbiaises)
            print 'Starting epoch ', epoch

            for uid in range(self.num_train) :

                activations = np.dot(X_train[uid], self.Wijk) + self.hidbiaises
                #if uid==0:print np.sum(activations)
                probs = np.apply_along_axis(logistic, 0, activations)
                states = probs > 0.5
                negdata[uid] = np.apply_along_axis(logistic,0,np.dot(self.Wijk, states) + self.visbiaises)
                #negdata[uid] = np.dot(self.Wijk, states) + self.visbiaises
                #negdata[uid] = np.apply_along_axis(logistic, 0, negdata[uid])
                #activations = np.dot(negdata[uid], self.Wijk) + self.hidbiaises
                negprobs = np.apply_along_axis(logistic, 0, np.dot(negdata[uid], self.Wijk) + self.hidbiaises)
                for i in range(self.numdims) :
                    posprod = X_train[uid][i] * probs
                    negprod = negdata[uid][i] * negprobs
                    Wijk_inc[i,:] = momentum*Wijk_inc[i] + self.epsilonw * (posprod-negprod) - self.weightcost*self.epsilonw*self.Wijk[i]
                    visbiais_inc[i] = momentum*visbiais_inc[i] + self.epsilonvb * (X_train[uid][i] - negdata[uid][i]) - self.weightcost*self.epsilonvb*self.visbiaises[i]
                    hidbiais_inc = momentum*hidbiais_inc + self.epsilonhb * (probs - negprobs) - self.weightcost*self.epsilonhb*self.hidbiaises
                    self.Wijk[i,:] += Wijk_inc[i,:]
                    self.visbiaises[i] += visbiais_inc[i]
                    self.hidbiaises += hidbiais_inc
            #print np.sum(abs(Wijk_inc)),np.sum(abs(visbiais_inc)),np.sum(abs(hidbiais_inc))

            if self.l==2 :
                neg = self.rbms[0].predictBackward(X_initial,negdata)
            else :
                neg2 = self.rbms[1].predictBackward(negdata)
                neg = self.rbms[0].predictBackward(X_initial,neg2)

            for uid in range(self.num_train) :
                for i in range(len(X_initial[uid])) :
                    err += abs(f(X_initial[uid][i][1]) - f(neg[uid][i][1]))
                    cnt += 1

            self.epsilonw *= 0.95
            self.epsilonvb *= 0.95
            self.epsilonhb *= 0.95
            self.weightcost *= 0.95

            print 'Error :', err/cnt

        print 'Training complete .... '
        np.save('weights_50_125_200_items/l'+str(self.l)+'_weights.npy',self.Wijk)
    	np.save('weights_50_125_200_items/l'+str(self.l)+'_vis.npy',self.visbiaises)
    	np.save('weights_50_125_200_items/l'+str(self.l)+'_hid.npy',self.hidbiaises)

    def predictForward(self, X_train) :

        activations = np.dot(X_train, self.Wijk) + self.hidbiaises[np.newaxis, :]
        probs = np.apply_along_axis(logistic, 0, activations)
        states = probs > 0.5
        return states

    def predictBackward(self, states) :

        #negdata = np.dot(states, self.Wijk.T) + self.visbiaises[np.newaxis, :]
        negdata = np.apply_along_axis(logistic, 0, np.dot(states, self.Wijk.T) + self.visbiaises[np.newaxis, :])
        #negdata = negdata > 0.5
        return negdata
