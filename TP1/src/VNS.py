# step 1
# Find a initial solution by DFS

import numpy as np
import time

def read_graph():
    return np.loadtxt("montreal", dtype='i', delimiter=',')

graph = read_graph()

from random import shuffle, randint


import copy

class Solution:
    def __init__(self, graph, places):
        self.g=0                # cost function
        self.graph=graph
        self.last_visited=places[0]
        self.visited=[places[0]]
        self.not_visited=copy.deepcopy(places[1:])

    def add(self, idx):
        to_be_pushed=idx
        self.visited.append(to_be_pushed)          #add the place into visited set
        self.g += graph[self.last_visited][to_be_pushed] # update the cost function
        self.last_visited=to_be_pushed             #update the last visited
        self.not_visited.remove(to_be_pushed)                           # delete the places[idx] from not_visited set

def initial_sol(graph, places):
    sol=Solution(graph, places)
    return dfs(sol,graph, places)
def dfs(root_node, graph, places):
    assert(root_node.g==0)
    assert(root_node.last_visited==places[0])
    node_stack=[]
    node_stack.append(root_node)
    while(True):
        current_node=node_stack.pop()
        if(len(current_node.not_visited)==1):
            current_node.add(current_node.not_visited[0])
            return current_node
        i=0
        candidate_places = current_node.not_visited
        while(i<len(current_node.not_visited)-1):
            selected_idx=randint(0,len(candidate_places)-2)
            new_node = copy.deepcopy(current_node)
            next_place=candidate_places[selected_idx]
            new_node.add(next_place)
            node_stack.append(new_node)
            i+=1


def shaking(sol,k):
    i=0
    result=[]
    while(i<k):
        idx_1=0
        idx_2=0
        while(idx_1==idx_2):
            idx_1=randint(1,len(sol.visited)-2)
            idx_2=randint(1,len(sol.visited)-2)
        new_neighborhood=copy.deepcopy(sol)
        tmp= new_neighborhood.visited[idx_1]
        new_neighborhood.visited[idx_1]= new_neighborhood.visited[idx_2]
        new_neighborhood.visited[idx_2]=tmp
        new_neighborhood.g = 0
        for idx in range(len(new_neighborhood.visited)-1):
            new_neighborhood.g+=new_neighborhood.graph[new_neighborhood.visited[idx]][new_neighborhood.visited[idx+1]]
        result.append(new_neighborhood)
        i+=1
    return result


def local_search_2opt(sol):
    for i in range(1,len(sol.visited)-1):
        for j in range(1,len(sol.visited)-1):
            if (i!=j):
                candidate_sol=copy.deepcopy(sol)
                tmp=candidate_sol.visited[i]
                candidate_sol.visited[i]=candidate_sol.visited[j]
                candidate_sol.visited[j]=tmp
                candidate_sol.g=0
                for idx in range(len(candidate_sol.visited)-1):
                    candidate_sol.g+=candidate_sol.graph[candidate_sol.visited[idx]][candidate_sol.visited[idx+1]]
                if(candidate_sol.g<sol.g):
                    sol=candidate_sol
    return sol

def VNS(sol,k_max,t_max):
    elapsed_time=0
    best_sol=sol
    while(True):
        k_neighborhood=shaking(best_sol,k_max)
        for k_sol in k_neighborhood:
            start_time = time.time()
            candidate=local_search_2opt(k_sol)
            if(best_sol.g>candidate.g):
                best_sol=candidate
            elapsed_time+=time.time()-start_time
            if(elapsed_time>t_max):
                break
        if (elapsed_time > t_max):
            break
    return best_sol

#test 1  --------------  OPT. SOL. = 27
start_time = time.time()
places=[0, 2, 20, 3, 18, 12, 13, 5, 11, 16, 15, 4, 9, 14, 1]
sol = initial_sol(graph=graph, places=places)
print(sol.visited)
print(sol.g)
result=VNS(sol,10,1)
print("after local search:\n")
print(result.visited)
print(result.g)
sum=0
for i in range(len(result.visited)-1):
    sum+=graph[result.visited[i]][result.visited[i+1]]
print (sum)
#------------------------------------
print("--- %s seconds ---" % (time.time() - start_time))

