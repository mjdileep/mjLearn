#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : softmax_regressor
# @Date : 2017-05-20-08-02
# @Poject: softmax
# @AUTHOR : Jayamal M.D.

'''NOTE THAT THIS MULTI-CLASS CLASSIFICATION ALGORITHM ASSUMES THAT THE DATASET [X]
CAN BE EXPRESSED AS AN EXPONENTIAL FAMILY DISTRIBUTION
AS WELL AS THE THE DATASET[X] AND THE TRAGETSET[Y] ARE LINEARLY RELATED!'''

import numpy as np


class softMaxRegressor:
    def __init__(self, alpha=0.01, max_itr=1000, plot=True):
        self.alpha = alpha
        self.max_itr = max_itr
        self.plot = plot

    def train(self, data, targets, alpha=0.001, max_itr=1000, plot=True, bandwidth=10):
        self.alpha = alpha
        self.max_itr = max_itr
        self.bandwidth = bandwidth
        self.plot = plot
        self.X = np.matrix(data)
        self.Y = np.matrix(targets)
        self.min_matrix = np.zeros(self.X.shape[1])
        self.max_matrix = np.zeros(self.X.shape[1])
        self.X = self.init_normalize(self.X)
        self.target_vec_size = self.Y.shape[1]
        self.input_vec_size = self.X.shape[1]
        self.teta = np.random.random((self.target_vec_size, self.input_vec_size))
        for itr in range(self.max_itr):
            for i in range(self.X.shape[0]):
                xi = self.X[i]
                devisor = self.calc_devisor(xi)
                yi = self.Y[i].getA1()
                target = 0
                axis = 0
                for j in range(yi.shape[0]):
                    if yi[j] > target:
                        target = yi[j]
                        axis = j
                if target < 0.5:
                    continue
                j = axis
                # print(str(j)+":",xi.getA1())
                upper_derivative = np.exp(np.dot(self.teta[j], xi.getA1()))
                derivative = upper_derivative / devisor
                # print(derivative)
                derivative = (1 - derivative) * xi
                # print(str(j)+","+str(derivative)+",teta:"+str(self.teta[j]))
                self.teta[j] = self.teta[j] + self.alpha * derivative

    def init_normalize(self, M):
        M = M.transpose() * 1.0
        for i in range(M.shape[0]):
            self.min_matrix[i] = min(M[i].getA1())
            self.max_matrix[i] = max(M[i].getA1())
            div = self.max_matrix[i] - self.min_matrix[i]
            M[i] = M[i] - self.min_matrix[i]
            M[i] = M[i] / div
        return M.transpose() * self.bandwidth

    def normalize(self, x):
        x = x.transpose() * 1.0
        for i in range(x.shape[0]):
            div = self.max_matrix[i] - self.min_matrix[i]
            x[i] = x[i] - self.min_matrix[i]
            x[i] = x[i] / div
        return x.transpose() * self.bandwidth

    def calc_liklyhood(self, X, teta):

        return 0

    def calc_devisor(self, x):
        val = 0
        for tetaj in self.teta:
            val += np.exp(np.dot(tetaj, x.getA1()))
        return val

    def calc_htetax(self, x):
        # x is not a np array

        htetax = np.zeros(self.teta.shape[0])
        devisor = self.calc_devisor(x)

        for i in range(self.teta.shape[0] - 1):
            # print(np.dot(self.teta[i],x.getA1()),self.teta[i],x.getA1())
            htetax[i] = np.exp(np.dot(self.teta[i], x.getA1())) / devisor
        htetax[self.teta.shape[0] - 1] = 1 - sum(htetax)

        return htetax

    def partial_train(self):
        pass

    def predict(self, X):
        result = []
        for each in X:
            x = self.normalize(each)
            prediction=self.calc_htetax(x)
            temp=np.zeros(prediction.shape[0]).astype(int)
            temp[np.argmax(prediction)] = 1
            result.append(temp)

        return result
