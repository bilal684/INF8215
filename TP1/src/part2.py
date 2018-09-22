# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:43:22 2018

@author: Bilal Itani, Mohammed Esseddik Ben Yahia, Xiangyi Zhang
"""

import numpy as np
import time
import copy
import heapq
import sys

class Vertex:
    visitedVertexes = []
    def __init__(self):
        self.id = None
        self.currentCost = sys.maxsize
        self.parent = None

    def __lt__(self, other):
        return self.currentCost < other.currentCost


def fastest_path_estimation(sol):
    """
    Returns the time spent on the fastest path between 
    the current vertex c and the ending vertex pm
    """
    c = sol.visited[-1]
    pm = sol.not_visited[-1]

    graph = read_graph()
    heap = []
    heapq.heapify(heap)
    currentVertex = Vertex()
    currentVertex.id = c
    currentVertex.currentCost = 0
    heapq.heappush(heap, currentVertex)
    while len(heap) > 0:
        currentVertex = heapq.heappop(heap)
        if currentVertex.id == pm: #We're done.
            break
        for i in range(0, len(graph[currentVertex.id])):
            if graph[currentVertex.id, i] > 0 and i not in Vertex.visitedVertexes:
                adjacentVertex = Vertex()
                adjacentVertex.id = i
                newCost = currentVertex.currentCost + graph[currentVertex.id, i]
                if newCost < adjacentVertex.currentCost:
                    adjacentVertex.currentCost = newCost
                    adjacentVertex.parent = currentVertex
                    heapq.heappush(heap, adjacentVertex)
        Vertex.visitedVertexes.append(currentVertex.id)
    return currentVertex.currentCost

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
        print("Cost = " + str(bestSol.g + bestSol.h))
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
        # self.not_visited = copy.deepcopy(places[1:]) # list of attractions not yet visited
        
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
        return self.g + self.h < other.g + other.h
    

        

def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')


graph = read_graph()
#test 3  --------------  OPT. SOL. = 26
start_time = time.time()
places=[0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
astar_sol = A_star(graph=graph, places=places)
print(astar_sol.g)
print(astar_sol.visited)
print("--- %s seconds ---" % (time.time() - start_time))
