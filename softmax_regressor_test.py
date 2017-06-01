#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#@Filename : softmax_regressor_test
#@Date : 2017-06-01-06-28
#@Poject: mjLearn
#@AUTHOR : Jayamal M.D.
import  numpy  as np
from lib import softmax_regressor as sr

X=np.matrix([[4,2,3],[7,7,3],[2,12,4],[9,10,2],[7,16,9],[5,13,9],[2,25,15],[2,22,18]])
Y=np.matrix([[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],[0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]])
var=sr.softMaxRegressor()
var.train(X,Y,bandwidth=10,alpha=0.001,max_itr=1000)
test=np.matrix([[2,12,4],[4,3,3]])
rs=var.predict(test)
for i in rs:
    print(i)