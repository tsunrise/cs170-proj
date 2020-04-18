from typing import List
import networkx as nx
from parse import read_input_file, write_output_file
from utils import average_pairwise_distance_fast, is_valid_network, average_pairwise_distance
import random
import sys

# TYPE DEFINITION

nxGraph = nx.classes.Graph

def randomDominatingTree(G: nxGraph) -> nxGraph:
    """
    Generate a random dominating tree basing on heuristics. 
    """

    pass

class EmployedBee:

    def __init__(self, T: nxGraph, G: nxGraph):
        self.solution: nxGraph = T
        self.G: nxGraph = G
        self.unimprovedTimes: int = 0
        self.currentCost: float = average_pairwise_distance_fast(self.solution)

    def scout(self) -> None:
        self.solution = randomDominatingTree(self.G)
        self.unimprovedTimes = 0
        self.currentCost = average_pairwise_distance_fast(self.solution)

    def work(self) -> bool: # find neighbor
        """
        Try to Find a neighbor solution. Return true if the solution improves (cost goes down). 
        """

        # TODO: ADD CODE HERE (find_neighbor)
        ...
        # TODO: ADD CODE HERE (update current cost)


def ABC(G: nxGraph, n_employed: int, n_onlooker:int, n_iter: int, fire_limit: int) -> nxGraph:
    """
    The artificial bee algorithm. Return an approximate connected dominating tree with minimum routing cost. 
    n_iter: the total number of iterations
    fire_limit: the maximum number of iterations that a bee does not improve its solution (if the limit is exceeded, the bee scouts)
    """
    
    # initialize employed bees
    bees: List[EmployedBee] = []
    isBeeImproved: List[bool]= [False] * n_employed

    for i in range(n_employed):
        T = randomDominatingTree(G)
        bees.append(EmployedBee(T, G))



    # Iteration stage
    for curr_iter in range(n_iter):
        # Each employed bee calls find_neighbor to try to find a solution toward its local optimum.
        for index, bee in enumerate(bees):
            improved = bee.work() # employ(S)
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
        if bees[first_random_index].currentCost > bees[second_random_index].currentCost:
            selectedBee = bees[second_random_index]
            selectedIndex = second_random_index

        improved = selectedBee.work()
        if improved:
            isBeeImproved[selectedIndex] = True

        
        # If an employed bee solution is not improved over time
        # fire the bee (let the bee find a new solution from scratch).
        for index, bee in enumerate(bees):
            if isBeeImproved[index]:
                bee.unimprovedTimes = 0
            else:
                bee.unimprovedTimes += 1

            if bee.unimprovedTimes > fire_limit:
                bee.scout()
        
    # Final Decision
    bestBee = min(bees, key=lambda b: b.currentCost)
    return bestBee.solution



def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!
    pass


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
