import numpy as np
import copy


class Regressor():
    def __init__(self, x=None, y=None, alpha=0.01, max_itr=10000, target_error=0.005, fit_function="GD"):
        if x is None or y is None:
            self.mat = None
            self.x = None
            self.y = None
            self.alpha = alpha
            self.max_itr = max_itr
            self.target_error = target_error
            self.fit_function = fit_function
            return

        if len(x) != len(y):
            raise "Length of inputs and targets doesn't match!(" + len(x) + " vs " + len(y) + ")"
        ln = len(x[0])
        for each in x:
            if ln != len(each):
                raise "length of input parameters are not equal!"
        self.x = x
        self.y = y
        self.alpha = alpha
        self.max_itr = max_itr
        self.target_error = target_error
        self.fit_function = fit_function
        self.mat = np.random.random((ln))
        if self.fit_function == "GD":
            gradient_decendent(self.mat, self.x, self.y, self.alpha, self.max_itr, self.target_error)
        else:
            gradient_decendent(self.mat, self.x, self.y, self.alpha, self.max_itr, self.target_error)

    def fit(self, x, y):
        if len(x) != len(y):
            raise "Length of inputs and targets doesn't match!(" + len(x) + " vs " + len(y) + ")"
        ln = len(x[0])
        for each in x:
            if ln != len(each):
                raise "length of input parameters are not equal!"
        self.x = x
        self.y = y
        self.mat = np.random.random((ln))
        if self.fit_function == "GD":
            gradient_decendent(self.mat, self.x, self.y, self.alpha, self.max_itr, self.target_error)
        else:
            gradient_decendent(self.mat, self.x, self.y, self.alpha, self.max_itr, self.target_error)

    def partial_fit(self, x, y):

        if self.x is None or self.y is None:
            self.fit(x, y)
        else:
            if len(x) != len(y):
                raise "Length of inputs and targets doesn't match!(" + len(x) + " vs " + len(y) + ")"
            ln = len(self.x[0])
            for each in x:
                if ln != len(each):
                    raise "length of input parameters are not equal!"
            self.x += x
            self.y += y
            if self.fit_function == "GD":
                gradient_decendent(self.mat, self.x, self.y, self.alpha, self.max_itr, self.target_error)
            else:
                gradient_decendent(self.mat, self.x, self.y, self.alpha, self.max_itr, self.target_error)

    def predict(self, x):
        result = []
        for each in x:
            result.append(np.dot(self.mat, each))
        return result

    def nomarlize(self, x=None):
        if x is not None:
            for i in range(len(x)):
                for j in range(len(x[i])):
                    x[i][j] = (x[i][j] - self.mn) / (self.mx - self.mn)
            return x

        self.mx = max(self.y)
        self.mn = min(self.y)
        for i in range(len(self.x)):
            mn = min(self.x[i])
            mx = max(self.x[i])
            if self.mx < mx:
                self.mx = mx
            if self.mn < mn:
                self.mn = mn

        self.Y = copy.deepcopy(self.y)
        self.X = copy.deepcopy(self.x)
        for i in range(len(self.x)):
            self.Y[i] = (self.y[i] - self.mn) / (self.mx - self.mn)
            for j in range(len(self.x[i])):
                self.X[i][j] = (self.x[i][j] - self.mn) / (self.mx - self.mn)


def gradient_decendent(mat, x, y, alpha, max_itr, target_error):
    def calcError():
        error = 0
        ln = len(y)
        for i in range(ln):
            error += pow(np.dot(mat, x[i]) - y[i], 2)
        return error / 2

    def calcDerivativesOfErrorFnction(i, j):
        der = (np.dot(mat, x[i]) - y[i]) * x[i][j]
        return der

    for l in range(max_itr):
        if l % 10 == 0:
            e = calcError()
            if target_error > e:
                return mat

        for i in range(len(x)):
            for j in range(len(mat)):
                mat[j] -= alpha * calcDerivativesOfErrorFnction(i, j)
    return mat
