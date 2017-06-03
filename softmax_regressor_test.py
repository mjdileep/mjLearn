#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : softmax_regressor_test
# @Date : 2017-06-01-06-28
# @Poject: mjLearn
# @AUTHOR : Jayamal M.D.

'''NOTE THAT THIS MULTI-CLASS CLASSIFICATION ALGORITHM ASSUMES THAT THE DATASET [X]
CAN BE EXPRESSED AS AN EXPONENTIAL FAMILY DISTRIBUTION
AS WELL AS THE THE DATASET[X] AND THE TRAGETSET[Y] ARE LINEARLY RELATED!'''

import numpy  as np
from lib import softmax_regressor as sr

# X is the input set while Y is the target for the training set
X = np.matrix([[4, 2, 3], [7, 7, 3], [2, 12, 4], [9, 10, 2], [7, 16, 9], [5, 13, 9], [2, 25, 15], [2, 22, 18]])
Y = np.matrix(
    [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])

'''Here [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1] are four different output classes as the targets
and [4,2,3],[2,12,4].... are 3-dimentional inputs'''

var = sr.softMaxRegressor()
var.train(X, Y, bandwidth=10, alpha=0.001,
          max_itr=1000)  # here you have to keep the balance between bandwidth,alpha and max_itr

test = np.matrix([[2, 12, 4], [4, 3, 3]])  # test input

rs = var.predict(test)  # predict the results

for i in rs:
    print(i)
