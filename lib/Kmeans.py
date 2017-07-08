#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : Kmeans
# @Date : 2017-07-08-20-53
# @Poject: k-means
# @AUTHOR : Jayamal M.D.

import random, math
from scipy.spatial import distance as euc
from numpy import sum


class Kmeans:
    def find(self, set, num_of_clusters, clusters_heads=None, max_itr=1000):
        self.set = set
        self.clusters_heads = clusters_heads
        self.max_itr = max_itr
        self.num_of_clusters = num_of_clusters
        self.clusters = {}
        self.error_prev = math.inf

        if self.clusters_heads is None:
            for each in range(self.num_of_clusters):
                self.clusters["c" + str(each)] = {}
                self.clusters["c" + str(each)]["head"] = random.sample(self.set, 1)[0]
        while self.max_itr:
            for each in range(self.num_of_clusters):
                self.clusters["c" + str(each)]["set"] = []
            for each in self.set:
                cluster = self.getCluster(each)
                self.clusters[cluster]["set"].append(each)
            for each in self.clusters.keys():
                self.clusters[each]["head"] = self.getNewHead(self.clusters[each])

            error = self.calcError()
            if self.error_prev == error:
                return self.clusters
            else:
                self.error_prev = error

            self.max_itr -= 1

    def getCluster(self, item):
        distance = math.inf
        cluster = None
        for each in self.clusters.keys():
            d = self.getDistance(item, self.clusters[each]["head"])
            if d < distance:
                distance = d
                cluster = each
        return cluster

    def getDistance(self, item1, item2):
        try:
            return euc.euclidean(item1, item2)
        except:
            return 0

    def getNewHead(self, cluster):
        set = cluster["set"]
        size = len(set)
        try:
            return list(sum(set, axis=0) / size)
        except:
            return None

    def calcError(self):
        error = 0
        for each in self.clusters:
            for elm in self.clusters[each]["set"]:
                error += self.getDistance(elm, self.clusters[each]["head"])
        return error
