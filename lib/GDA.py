#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : GDA
# @Date : 2017-06-02-07-58
# @Poject: mjLearn
# @AUTHOR : Jayamal M.D.

'''NOTE THAT GDA( Gaussian Discriminant Analysis) IS A BINARY CLASSIFICATION ALGORITHM AND ASSUMES
YOUR DATASET UNDER EACH CLASS CAN BE EXPRESSED AS A GAUSSIAN DISTRIBUTION...'''

import numpy as np


class GDA_classifier:
    def __init__(self):
        self.X = None
        self.Y = None
        self.mean_0 = None
        self.mean_1 = None
        self.cov = None
        self.devisor = None

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        self.mean_0 = np.matrix(np.zeros((self.X.shape[1], 1)))
        self.mean_1 = np.matrix(np.zeros((self.X.shape[1], 1)))
        self.calc_means()
        self.calc_corvariance()
        self.calc_devisor()
        # print(self.mean_0,self.mean_1)

    def partial_train(self, X, Y):
        pass

    def predict(self, X):
        results = []
        for i in range(X.shape[0]):
            x = X[i]
            p0 = np.exp(-.5 * (
            np.dot(np.dot(x - self.mean_0.transpose(), np.linalg.inv(self.cov)), x.transpose() - self.mean_0)))
            p0 *= self.devisor
            p1 = np.exp(-.5 * (
            np.dot(np.dot(x - self.mean_1.transpose(), np.linalg.inv(self.cov)), x.transpose() - self.mean_1)))
            p1 *= self.devisor
            # print(str(p0)+","+str(p1))
            results.append([0 if p0 > p1 else 1])
        return np.matrix(results)

    def calc_devisor(self):
        val = (pow(2 * np.pi, self.X.shape[1] / 2)) * np.sqrt(np.linalg.det(self.cov))
        self.devisor = 1 / val

    def calc_means(self):
        count_0 = 0
        count_1 = 0
        for i in range(self.X.shape[1]):
            for j in range(self.X.shape[0]):
                if self.Y[j]:
                    self.mean_1[i] += self.X[j].transpose()[i]
                    count_1 += 1
                else:
                    self.mean_0[i] += self.X[j].transpose()[i]
                    count_0 += 1
        count_1 /= self.X.shape[1]
        count_0 /= self.X.shape[1]
        self.mean_0 /= count_0
        self.mean_1 /= count_1
        return self.mean_1, self.mean_0

    def calc_corvariance(self):
        var = np.zeros((self.X.shape[1], self.X.shape[1]))
        for i in range(self.X.shape[0]):
            if self.Y[i]:
                for j in range(self.X.shape[1]):
                    for k in range(self.X.shape[1]):
                        var[j][k] += np.dot((self.X[j] - self.mean_1.transpose()),
                                            (self.X[k].transpose() - self.mean_1))
            else:
                for j in range(self.X.shape[1]):
                    for k in range(self.X.shape[1]):
                        var[j][k] += np.dot((self.X[j] - self.mean_0.transpose()),
                                            (self.X[k].transpose() - self.mean_0))
        self.cov = np.matrix(var / self.X.shape[0])

    def calc_pi(self):
        sum = 0
        for i in range(self.Y.shape[0]):
            sum += self.Y[i].getA1()[0]
        return sum / self.Y.shape[0]
