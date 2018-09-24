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
    pm = sol.not_visited[-1]

    graph = sol.graph
    heap = []
    currentSol = copy.deepcopy(sol)
    currentSol.g = 0
    heapq.heapify(heap)
    heapq.heappush(heap, currentSol)
    while len(heap) > 0:
        currentSol = heapq.heappop(heap)
        if currentSol.visited[-1] == pm: #We're done.
            break
        for i in range(0, len(currentSol.not_visited)):
            if graph[currentSol.visited[-1], currentSol.not_visited[i]] > 0:
                newSol = copy.deepcopy(currentSol)
                newSol.addForDijkstra(i)
                heapq.heappush(heap, newSol)
    return currentSol.g

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

    def addForDijkstra(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        self.g += graph[self.visited[-1], self.not_visited[idx]]
        self.visited.append(self.not_visited.pop(idx))

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
