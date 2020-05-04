#!/usr/bin/env python3 

from typing import List, Tuple, Dict
import networkx as nx
from parse import read_input_file, read_output_file, write_output_file
from utils import average_pairwise_distance_fast, is_valid_network, average_pairwise_distance
import random
import sys
import solver
import multiprocessing
import os
import json

solver.RANDOMIZED_WEIGHT_VARIATION = 0.37 # graph randomized level
N_EMPLOYED = 20 # number of employed bees
N_ONLOOKER = 5 # number of onlooker bees
N_ITERATIONS = 50000 # number of iterations of ABC
FIRE_LIMIT = 100 # maximum iterations allowed for a bee to not discover a better option
TERMINATION_LIMIT = 5000 # the maximum number of iterations allowed for the whole not improve its solution

TEAM_NAME = 'Salieri'
def solveFile(fileName: str, log = False) -> bool:
    """
    Solve a graph saved in ./inputs/{fileName}.in and output it in output folder. 
    Return if solve file succeed. 
    """

    try:
        G = read_input_file("./inputs/%s.in" % fileName)
        T = solver.ABC(G, N_EMPLOYED, N_ONLOOKER, N_ITERATIONS, FIRE_LIMIT, TERMINATION_LIMIT, log = log)
        assert(is_valid_network(G, T))

        if os.path.exists("./outputs/%s.out" % fileName):
            oldT = read_output_file("./outputs/%s.out" % fileName, G)
            if len(T) == 1 and len(oldT) != 1:
                write_output_file(T, "./outputs/%s.out" % fileName)
                return True
            if len(oldT) == 1 or average_pairwise_distance(oldT) <= average_pairwise_distance(T):
                # do nothing
                print("File %s is skipped because old tree is better. " % fileName)
                return True

        write_output_file(T, "./outputs/%s.out" % fileName)
        return True

    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        # stdout
        print("ERROR: An error occured when processing on %s: %s" % (fileName, sys.exc_info()[0]))
        return False

# selective processing (based on online rank)
def getNonFirst():
    data = None
    with open("rank.json","r") as f:
        data = json.load(f)
    for v in data.values():
        v.sort(key = lambda x: x[1])
    def getRank(value):
        curr_cost = -1
        for v in value:
            if v[0] == TEAM_NAME:
                curr_cost = v[1]
        rank = 1
        for v in value:
            if v[0] != TEAM_NAME and v[1] < curr_cost:
                rank += 1
        
        return rank
    stat = dict()
    for k, v in data.items():
        rank = getRank(v)
        if rank in stat:
            stat[rank].append(k)
        else:
            stat[rank] = [k]

    keys = sorted(stat.keys(), reverse = True)
    values = []
    for key in keys:
        if key != 1:
            values.extend(stat[key])
    return values


if __name__ == "__main__":
    # online algo until terminated manually
    print("Solver version: %s" % solver.VERSION)

    parallel = False
    runAll = False
    num_cores = multiprocessing.cpu_count()
    if len(sys.argv) >= 2:
        if sys.argv[1] == "-p" or sys.argv[1] == "--parallel":
            parallel = True
        if sys.argv[1] == "-a" or sys.argv[1] == "--all":
            runAll = True

    
    if len(sys.argv) >= 3:
        if sys.argv[2] == "-a" or sys.argv[2] == "--all":
            runAll = True


    if runAll:
        tasks = [f[:-3] for f in os.listdir("./inputs/") if f[-3:] == ".in"]
    else:
        try:
            tasks = getNonFirst()
        except:
            print("Please place rank.json in the same directory. Read README for more details. ")
            exit()
    
    # read number of tasks
    print("Program will perform those tasks: %s" % str(tasks))
    running_round = 1
    while not os.path.exists("./terminate.flag"):

        if not parallel:
            count = 0
            failure = []
            for task in tasks:
                count += 1
                print("Solving: %s (%d/%d)" % (task, count, len(tasks)))
                success = solveFile(task, log=True)
                if not success:
                    failure.append(task)
            if len(failure) != 0:
                print("%d/%d files are not solved successfully. Please check. " % (len(failure), len(tasks)))
                for f in failure:
                    print(f)
        else:
            print("Using %d subprocesses" % num_cores)
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
                    if os.path.exists("./terminate.flag"):
                        return
                if len(failure) != 0:
                    print("Thread %d reports that %d/%d files are not solved successfully. Please check. " % (thread_num, len(failure), len(subtasks)))
                    for f in failure:
                        print("FAIL: " + f)
                else:
                    print("Thread %d complete. " % thread_num)
            
            # task randomization
            random.shuffle(tasks)

            threads:List[multiprocessing.Process] = []
            for i in range(num_cores):
                t = multiprocessing.Process(target=thread_task, args=(i, num_cores, tasks))
                threads.append(t)
                t.start()
            for i in range(num_cores):
                threads[i].join()
            print("Round %d complete" % running_round)
        running_round += 1
    print("Program terminated because the program found terminate.flag")
                
        




            

