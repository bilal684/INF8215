# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:43:22 2018

@author: Bilal Itani, Mohammed Esseddik Ben Yahia, Xiangyi Zhang
"""

import numpy as np
import time
import copy
import heapq


def fastest_path_estimation(sol):
    """
    Returns the time spent on the fastest path between 
    the current vertex c and the ending vertex pm
    """
    c = sol.visited[-1]
    pm = sol.not_visited[-1]
    heap = []
    heapq.heapify(heap)
    for i in range(0, len(sol.not_visited)):
        heapq.heappush(heap, (sol.graph[c, sol.not_visited[i]], sol.not_visited[i]))
    lowestCostToPm = sol.graph[c, pm]
    while len(heap) > 0:
        firstTuple = heapq.heappop(heap)
        if firstTuple[1] == pm:
            lowestCostToPm = firstTuple[0]
            break
        for i in range(0, len(heap)):
            if firstTuple[0] + sol.graph[firstTuple[1], heap[i][1]] < heap[i][0]:
                newTuple = (firstTuple[0] + sol.graph[firstTuple[1], heap[i][1]], heap[i][1])
                heap.remove((heap[i][0], heap[i][1]))
                heapq.heapify(heap)
                heapq.heappush(heap,newTuple)

    return lowestCostToPm

def A_star(graph, places):
    """
    Performs the A* algorithm
    """

    # blank solution
    root = Solution(graph=graph, places=places)
    # search tree T
    T = []
    heapq.heapify(T)
    heapq.heappush(T, root)
    bestSol = None
    while len(T) > 0:
        bestSol = heapq.heappop(T)
        if len(bestSol.not_visited) == 0:
            break
        for i in range(0, len(bestSol.not_visited) - 1):
            newSol = copy.deepcopy(bestSol)
            newSol.add(i)
            heapq.heappush(T, newSol)
        if len(bestSol.not_visited) == 1:
            newSol = copy.deepcopy(bestSol)
            newSol.add(0)
            heapq.heappush(T, newSol)
    return bestSol




class Solution:
    childsCount = 0
       
    def __init__(self, places, graph):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0 # current cost
        self.graph = graph 
        self.visited = [places[0]] # list of already visited attractions
        self.not_visited = copy.deepcopy(places[1:])
        self.h = 0 # Estimated cost
        
    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        self.g += graph[self.visited[-1], self.not_visited[idx]]
        self.visited.append(self.not_visited.pop(idx))
        if len(self.not_visited) > 0:
            self.h = fastest_path_estimation(self)
        else:
            self.h = 0

    def __lt__(self, other):
        if self.g + self.h == other.g + other.h:
            if len (self.visited) > len(other.visited):
                return True
            else:
                return False
        return self.g + self.h < other.g + other.h
    

        

def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')


graph = read_graph()
#test 3  --------------  OPT. SOL. = 26
start_time = time.time()
places=[0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
astar_sol = A_star(graph=graph, places=places)
print(astar_sol.g)
print(astar_sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))
