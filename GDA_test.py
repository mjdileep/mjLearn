#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : GDA_test
# @Date : 2017-06-03-09-04
# @Poject: mjLearn
# @AUTHOR : Jayamal M.D.
'''NOTE THAT GDA( Gaussian Discriminant Analysis) IS A BINARY CLASSIFICATION ALGORITHM AND ASSUMES
YOUR DATASET UNDER EACH CLASS CAN BE EXPRESSED AS A GAUSSIAN DISTRIBUTION...'''

from  lib import GDA
import numpy as np

var = GDA.GDA_classifier()

# define the training set
x = np.matrix([[2, 7, 9], [1, 1, 2], [3, 10, 6], [2, 7, 7], [1, 2, 1], [3, 8, 9]])
y = np.matrix([[1], [0], [1], [1], [0], [1]])

# train the model
var.train(x, y)

# test with some data
test = np.matrix([[20, 22, 20], [0, 2, 0], [2, 5, 5]])
rs = var.predict(test)

print(rs)
