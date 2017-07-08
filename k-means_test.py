#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : k-means_test
# @Date : 2017-07-08-22-42
# @Poject: mjLearn
# @AUTHOR : Jayamal M.D.
from lib import Kmeans

var = Kmeans.Kmeans()
print(var.find(set=[[1, 2, 3], [1, 2, 2], [2, 1, 3], [3, 4, 5], [2, 4, 6], [6, 1, 3], [1, 7, 2], [5, 4, 8], [3, 4, 7]],
               num_of_clusters=2))
