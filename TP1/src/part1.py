# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:43:22 2018

@author: Bilal Itani, Mohammed Esseddik Ben Yahia, Xiangyi Zhang
"""

import numpy as np
import time
import copy

from queue import Queue

def bfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """
    solutionInitial = Solution(places, graph)
    queue = Queue()
    queue.put(solutionInitial)
    bestSolution = None
    while queue.qsize() > 0:
        sol = queue.get()
        if len(sol.not_visited) == 0:   
            if bestSolution is None:
                bestSolution = sol
            elif bestSolution.g > sol.g:
                bestSolution = sol
        i = 0
        while i < len(sol.not_visited) - 1:
            newSolution = copy.deepcopy(sol)
            newSolution.add(i)
            queue.put(newSolution)
            i+=1
        if len(sol.not_visited) == 1:
            newSolution = copy.deepcopy(sol)
            newSolution.add(0)
            queue.put(newSolution)
    return bestSolution

class Node:
    def __init__(self, solution):
        self.solution = solution
        self.childs = []

    def addChild(self, Node):
        self.childs.append(Node)

    #def isGoal():
        
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
        
    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        self.g += graph[self.visited[-1], self.not_visited[idx]]
        self.visited.append(self.not_visited.pop(idx))        

def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')

graph = read_graph()
 
#test 1  --------------  OPT. SOL. = 27
start_time = time.time()
places=[0, 5, 13, 16, 6, 9, 4]
sol = bfs(graph=graph, places=places)
print(sol.g)
print("--- %s seconds ---" % (time.time() - start_time))
