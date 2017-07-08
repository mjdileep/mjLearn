#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : linear_regressor_test
# @Date : 2017-07-08-22-42
# @Poject: mjLearn
# @AUTHOR : Jayamal M.D.
import numpy as np
from lib import linear_regressor

X = []
Y = []
for i in range(10):
    x1 = np.random.randint(1, 10)
    x2 = np.random.randint(1, 10)
    y = (x1 / 7 + x2 / 3) / 9 - 2 * x1 + 5 * x2
    X.append([x1, x2])
    Y.append(y)


print(X)
print(Y)
v = linear_regressor.Regressor()
v.fit(X, Y)
print(v.predict([[5, 9]]))
