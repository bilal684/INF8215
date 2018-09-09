# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 10:43:22 2018

@author: Bilal Itani
"""

import numpy as np
import time
import copy

#from queue import Queue

def bfs(graph, places):
    """
    Returns the best solution which spans over all attractions indicated in 'places'
    """
    solution = Solution(places, graph)
    root = Node(solution)
    buildTree(root)
    print(Solution.childsCount)
    
def buildTree(parent):
    if(len(parent.solution.not_visited) > 0):
        i = 0
        while i < len(parent.solution.not_visited) - 1:
            newNode = Node()
            newSolution = copy.deepcopy(parent.solution)
            newSolution.add(i)
            newNode.solution = newSolution
            parent.addChild(buildTree(newNode))
            i+=1
        if len(parent.solution.not_visited) == 1:
            newNode = Node()
            newSolution = copy.deepcopy(parent.solution)
            newSolution.add(i)
            newNode.solution = newSolution
            parent.addChild(newNode)
        return parent
    return None

class Node:
    def __init__(self, solution = None):
        self.solution = solution
        self.childs = []
    
    def addChild(self, Node):
        self.childs.append(Node)
        
class Solution:
    childsCount = 0
       
    def __init__(self, places=None, graph=None):
        """
        places: a list containing the indices of attractions to visit
        p1 = places[0]
        pm = places[-1]
        """
        self.g = 0 # current cost
        self.graph = graph 
        if(places is None):
            self.visited = []
            self.not_visited = []
        else:
            self.visited = [places[0]] # list of already visited attractions
            self.not_visited = copy.deepcopy(places[1:])
        #self.not_visited = copy.deepcopy(places[1:]) # list of attractions not yet visited
        
    def add(self, idx):
        """
        Adds the point in position idx of not_visited list to the solution
        """
        self.g += graph[self.visited[-1], self.not_visited[idx]]
        self.visited.append(self.not_visited.pop(idx))
        Solution.childsCount = Solution.childsCount + 1
    

        

def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')

graph = read_graph()
 
#test 1  --------------  OPT. SOL. = 27
start_time = time.time()
places=[0, 2, 7, 13, 11, 16, 15, 7, 9, 8, 4]
sol = bfs(graph=graph, places=places)
print(Solution.childsCount)
#print(sol.g)
print("--- %s seconds ---" % (time.time() - start_time))
