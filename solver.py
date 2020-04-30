from typing import List, Tuple, Dict
import networkx as nx
from parse import read_input_file, write_output_file
from utils import average_pairwise_distance_fast, is_valid_network, average_pairwise_distance
import random
import sys
from networkx.algorithms import approximation

# TYPE DEFINITION

nxGraph = nx.classes.Graph

# CONSTANTS

VERSION = "regretV3"

RANDOMIZED_WEIGHT_VARIATION: float = 0.35
REGRET_PRUNE_RATE: float = 0.18
SPT_RATE: float = 0.2

def randomDominatingTree(G: nxGraph, init: bool = False) -> nxGraph:
    """
    Generate a random dominating tree basing on heuristics. 
    This algorithm randomly selects some strategies: randomized minimum spanning tree, randomized shortest path tree
    """

    # current baseline approach: 
    if init:
        rG = G.copy()
    else:
        rG = randomizedGraph(G, RANDOMIZED_WEIGHT_VARIATION)

    useMST: bool = random.choice([True, False])

    if useMST:
        return nx.minimum_spanning_tree(rG, weight='r_weight')
    else:
        # find a random starting vertex
        start: int = random.choice(list(G.nodes))
        paths = nx.single_source_dijkstra(rG, start, weight='r_weight')[1]
        nG = nx.Graph()
        nG.add_nodes_from(G.nodes)
        for dest in paths:
            path: List[int] = paths[dest]
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                if not nG.has_edge(a, b):
                    nG.add_edge(a, b)
                    nG[a][b]['weight'] = G[a][b]['weight']
        
        return nG


def randomizedGraph(G: nxGraph, variation: float, floor: float = 1e-3) -> nxGraph:
    """
    Generate a randomized graph. The weight is randomized in the way that the new weight of the edge is sample from 
    original weight * (1 +- Normal(0, variation))
    floor: the minimum weight of an edge allowed
    """

    nG: nxGraph = G.copy()
    for u, vs in nG.adjacency():
        for v in vs:
            nG[u][v]['r_weight'] = max(G[u][v]['weight'] * (1 + random.normalvariate(0, variation)), floor)
    
    return nG


class EmployedBee:

    def __init__(self, G: nxGraph, empty: bool = False):
        if empty:
            self.G = G
            return
        self.id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz1234567890") for i in range(8))
        self.solution: nxGraph = None
        self.G: nxGraph = G
        self.unimprovedTimes: int = 0
        self.leaves: List[int] = []
        self.regretTree: nxGraph = None  # when the program decides to prune a leaf even when it knows the cost will increase, the program store the previous tree
        self.regretLeaves: List[int] = []
        self.regretTreeCost: float = 0
        self.scout(init = True)

    def scout(self, init: bool = False) -> None:
        # has some chance to use best SPT
        if not init and random.random() < SPT_RATE:
            init = True
        self.solution = randomDominatingTree(self.G, init = init)
        self.unimprovedTimes = 0
        if len(self.solution) == 1:
            self.currentCost = 0
        else:
            self.currentCost = average_pairwise_distance_fast(self.solution)
        self.leaves = []
        
        # find leaves in the tree
        for v in self.solution.nodes:
            if len(self.solution[v]) == 1:
                self.leaves.append(v)

    def copy(self):
        b: EmployedBee = EmployedBee(self.G, empty = True)
        b.solution = self.solution.copy()
        b.unimprovedTimes = self.unimprovedTimes
        b.leaves = self.leaves.copy()
        b.currentCost = self.currentCost
        if self.regretTree is not None:
            b.regretTree = self.regretTree.copy()
        else:
            b.regretTree = None
        b.regretTreeCost = self.regretTreeCost
        b.regretLeaves = self.regretLeaves.copy()
        b.id = self.id
        
        return b

    def work(self) -> bool: # find neighbor
        """
        Try to Find a neighbor solution. Return true if the solution improves (cost goes down). 
        """

        # find_neighbor

        # try to randomly remove a leaf
        T = self.solution

        if (len(T) == 1):
            new_cost = 0
            self.currentCost = new_cost
            return True # early termination: solution cost  = 0

        toRemoveLeafIndex = random.randint(0, len(self.leaves) - 1)
        toRemove = self.leaves.pop(toRemoveLeafIndex)

        parent = list(T[toRemove])[0]

        edge_weight = T[parent][toRemove]['weight']
        T.remove_node(toRemove)

        if len(T) == 0 or not is_valid_network(self.G, T):
            # restore T and give up
            T.add_node(toRemove)
            T.add_edge(parent, toRemove, weight = edge_weight)
            self.leaves.append(toRemove)
            return False
        
        if (len(T) == 1):
            new_cost = 0
            self.currentCost = new_cost
            return True # early termination: solution cost  = 0
        else:
            new_cost = average_pairwise_distance_fast(T)
        if new_cost > self.currentCost:
            chance = random.random()
            if chance < REGRET_PRUNE_RATE and len(self.leaves) > 1:
                # do not restore, regret pruning (prune even if cost goes up)
                if self.regretTree is None or new_cost < self.regretTreeCost:
                    self.regretTree = T.copy()
                    self.regretTree.add_node(toRemove)
                    self.regretTree.add_edge(parent, toRemove, weight = edge_weight)
                    self.regretLeaves = self.leaves.copy()
                    self.regretLeaves.append(toRemove)
                    self.regretTreeCost = self.currentCost
                self.currentCost = new_cost
                if len(T[parent]) == 1:
                    self.leaves.append(parent)
                return False
            else:
                # restore T and give up
                T.add_node(toRemove)
                T.add_edge(parent, toRemove, weight = edge_weight)
                self.leaves.append(toRemove)
                return False
        
        # update success and add parent to leaves if possible
        self.currentCost = new_cost
        if len(T[parent]) == 1:
            self.leaves.append(parent)
        return True

    def getSolution(self) -> nxGraph:
        if self.regretTree is not None and self.regretTreeCost < self.currentCost:
            return self.regretTree
        return self.solution
    
    def getSolutionCost(self) -> float:
        if self.regretTree is not None and self.regretTreeCost < self.currentCost:
            return self.regretTreeCost
        return self.currentCost

    def restoreBest(self) -> bool:
        # restore to prior-prune state. No regret. 
        if self.regretTree is not None and self.currentCost > self.regretTreeCost:
            self.solution = self.regretTree.copy()
            self.currentCost = self.regretTreeCost
            self.regretTree = None
            self.leaves = self.regretLeaves.copy()
            self.regretLeaves = []
            return True

        return False
def zeroCostCheck(G: nxGraph) -> int:
    n = len(G)
    for v, dic in G.adjacency():
        if len(dic) == n-1:
            return v
    return -1


def ABC(G: nxGraph, n_employed: int, n_onlooker:int, n_iter: int, fire_limit: int, termination_limit: int = 10000,log: bool = False) -> nxGraph:
    """
    The artificial bee algorithm. Return an approximate connected dominating tree with minimum routing cost. 
    n_iter: the total number of iterations
    fire_limit: the maximum number of iterations that a bee does not improve its solution (if the limit is exceeded, the bee scouts)
    termination_limit: the maximum number of iterations allowed for the whole not improve its solution
    """
    
    # pre-sanity-check: zero-cost tree
    pre_master_v: int = zeroCostCheck(G)
    if pre_master_v != -1:
        if log:
            print("Tree has zero cost. All Iterations skipped. ")
        T = nxGraph()
        T.add_node(pre_master_v)
        return T


    # initialize employed bees
    bees: List[EmployedBee] = []
    isBeeImproved: List[bool]= [False] * n_employed
    isThisIterImproved: bool = False
    unimprovedCounter = 0

    for i in range(n_employed):
        bees.append(EmployedBee(G))
    bee_counter = len(bees) # for statistics only
    bestBee: EmployedBee = bees[0]
    # Iteration stage
    for curr_iter in range(n_iter):
        # Each employed bee calls find_neighbor to try to find a solution toward its local optimum.
        for index, bee in enumerate(bees):
            improved = bee.work() # employ(S)
            if bee.getSolutionCost() < bestBee.getSolutionCost():
                isThisIterImproved = True
                unimprovedCounter = 0
                bestBee = bee
            if improved:
                isBeeImproved[index] = True

        # Each onlooker bee randomly chooses two employed bee and choose the bee whose solution has lower cost. 
        # The one chosen is called find_neighbor again.
        first_random_index = random.randint(0, n_employed - 1)
        second_random_index = random.randint(0, n_employed - 1)
        while second_random_index == first_random_index:
            second_random_index = random.randint(0, n_employed - 1)

        selectedBee: EmployedBee = bees[first_random_index]
        selectedIndex: int = first_random_index
        if bees[first_random_index].currentCost > bees[second_random_index].currentCost: # use current working tree cost instead of regret cost
            selectedBee = bees[second_random_index]
            selectedIndex = second_random_index

        improved = selectedBee.work()
        if selectedBee.getSolutionCost() < bestBee.getSolutionCost():
            isThisIterImproved = True
            unimprovedCounter = 0
            bestBee = selectedBee
        if improved:
            isBeeImproved[selectedIndex] = True

        
        # If an employed bee solution is not improved over time
        # fire the bee (let the bee find a new solution from scratch).
        for index, bee in enumerate(bees):
            if isBeeImproved[index]:
                bee.unimprovedTimes = 0
            else:
                bee.unimprovedTimes += 1

            if bee.unimprovedTimes >= fire_limit // 2:
                restored = bee.restoreBest() # restore to regret tree state

            if bee.unimprovedTimes > fire_limit:
                if bee == bestBee:
                    bestBee = bestBee.copy()
                bee.scout()
                bee_counter += 1
            
            # reset bee improved list
            isBeeImproved[index] = False

        if not isThisIterImproved:
            unimprovedCounter += 1
            if unimprovedCounter > termination_limit:
                if log:
                    print("(END) At iteration %d, the best cost is %f (working: %f, regret: %f, bee_id: %s), %d scouts are called" % (curr_iter, bestBee.getSolutionCost(), bestBee.currentCost, bestBee.regretTreeCost if bestBee.regretTree is not None else -1, bestBee.id, bee_counter))
                return bestBee.solution
        
        isThisIterImproved = False
        if log and curr_iter % 100 == 0:
            print("At iteration %d, the best cost is %f (working: %f, regret: %f, bee_id: %s), %d scouts are called" % (curr_iter, bestBee.getSolutionCost(), bestBee.currentCost, bestBee.regretTreeCost if bestBee.regretTree is not None else -1, bestBee.id, bee_counter))
        if bestBee.getSolutionCost() - 0 < 1e-5:
            if log:
                print("Found a bee whose cost is zero! Program finished early at iteration %d. " % curr_iter)
            return bestBee.solution
    # Final Decision
    return bestBee.solution



def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    return ABC(G, 15, 3, 1400, 100) # for autograder purpose only


# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

# if __name__ == '__main__':
#     assert len(sys.argv) == 2
#     path = sys.argv[1]
#     G = read_input_file(path)
#     T = solve(G)
#     assert is_valid_network(G, T)
#     print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
#     write_output_file(T, 'out/test.out')
