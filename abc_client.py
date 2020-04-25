from typing import List, Tuple, Dict
import networkx as nx
from parse import read_input_file, write_output_file
from utils import average_pairwise_distance_fast, is_valid_network, average_pairwise_distance
import random
import sys
import solver
import multiprocessing
import os

solver.RANDOMIZED_WEIGHT_VARIATION = 0.35 # graph randomized level
N_EMPLOYED = 15 # number of employed bees
N_ONLOOKER = 3 # number of onlooker bees
N_ITERATIONS = 1400 # number of iterations of ABC
FIRE_LIMIT = 100 # maximum iterations allowed for a bee to not discover a better option

def solveFile(fileName: str) -> bool:
    """
    Solve a graph saved in ./inputs/{fileName}.in and output it in output folder. 
    Return if solve file succeed. 
    """

    try:
        G = read_input_file("./inputs/%s.in" % fileName)
        T = solver.ABC(G, N_EMPLOYED, N_ONLOOKER, N_ITERATIONS, FIRE_LIMIT)
        assert(is_valid_network(G, T))
        write_output_file(T, "./outputs/%s.out" % fileName)
        return True

    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        # stdout
        print("ERROR: An error occured when processing on %s" % fileName)
        return False


if __name__ == "__main__":
    parallel = False
    num_cores = multiprocessing.cpu_count()
    if len(sys.argv) == 2:
        if sys.argv[1] == "-p" or sys.argv[1] == "--parallel":
            parallel = True

    # read number of tasks
    tasks = [f[:-3] for f in os.listdir("./inputs/") if f[-3:] == ".in"]
    if not parallel:
        count = 0
        failure = []
        for task in tasks:
            count += 1
            # print("Solving: %s (%d/%d)" % (task, count, len(tasks)))
            success = solveFile(task)
            if not success:
                failure.append(task)
        if len(failure) != 0:
            print("%d/%d files are not solved successfully. Please check. " % (len(failure), len(tasks)))
            for f in failure:
                print(f)
    else:
        # parallel part
        def thread_task(thread_num: int, n: int, tasks: List[str]):
            """
            run a thread task: n is number of cores
            """

            print("Start Thread: %d/%d" % (thread_num, n))
            subtasks = []
            for i in range(thread_num, len(tasks), n):
                subtasks.append(tasks[i])
            count = 0
            failure = []
            for task in subtasks:
                count += 1
                # print("Thread %d is solving: %s (%d/%d)" % (thread_num, task, count, len(subtasks)))
                success = solveFile(task)
                if not success:
                    failure.append(task)
            if len(failure) != 0:
                print("Thread %d reports that %d/%d files are not solved successfully. Please check. " % (thread_num, len(failure), len(subtasks)))
                for f in failure:
                    print("FAIL: " + f)
            else:
                print("Thread %d complete. " % thread_num)
        
        threads:List[multiprocessing.Process] = []
        for i in range(num_cores):
            t = multiprocessing.Process(target=thread_task, args=(i, num_cores, tasks))
            threads.append(t)
            t.start()
        for i in range(num_cores):
            threads[i].join()
            
        




            

