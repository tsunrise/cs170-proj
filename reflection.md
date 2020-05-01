# CS170 Project Reflection

Tom Shen, Ziang Gao, Alan Zhu

## Algorithms

Our algorithm is based on artificial bee colony algorithms with some slight modification. This is a randomized online algorithm. 

First, let's declare some helper function, which is useful for the workflow: 

- `scout`: This function can quickly generate a random solution with relatively high cost. 
- `work`:  This function takes a solution and try to find one that is slightly better than this solution. 

Here are the main workflow: 

- **Create some number of workers.** Each worker use `scout` to get a random solution. 
- **Repeat the following procedures (infinite).** Track the best solution we got so far. 
  - Each worker has a solution at this time. Each work tries to call `work` to find a better solution. The worker is moving toward the local optimum. 
  - We randomly choose two workers, and choose the one who has a solution with lower cost. Let this worker call `work` again. 
  - If a worker has not improved its solution for a certain amount of iterations, the worker will call `scout` to go to another random solution. 
  - If the best solution is not improved over a certain amount of iterations, terminate the program. 
- After we manually terminate the program, the program will return the best solution that a worker ever observed. 

**How `scout` work**: 

- We use MST and SPT as the heuristics to create random graph. First, given graph $$G$$, we sample a graph $$G'$$ so that each $$\forall e' \in G', w(e') = w(e) * \mbox{Normal}(1, \sigma)$$ where $$e$$ is the corresponding edge in $$G$$ and $$\sigma$$ is configurable. We randomly select whether we want to use MST or SPT as heuristics. 
- If we use MST, we calculate MST on $$G'$$ and check which edges are in tree. Return those edges. 
- If we use SPT, we randomly select a starting point and run Dijkstra's. Return edges in the resulting SPT. 

**How `work` work**: 

- We use leaf pruning in this method. Program randomly selects a leaf in the tree. 
- Try to remove the leaf:
  - If after removal, the solution is no longer valid, do not remove this leaf. 
  - If after removal, the solution is valid but has higher cost, than the program will remove this leaf in a small probability. 
  - If after removal, the solution is valid and has lower cost, remove the leaf. 

### How we calculate multiple files concurrently: 

- Create 8 threads using python `multiprocessing` library. Assign input randomly to each thread. 
- For each thread, process each task serially. 

### Why we think this algorithm is good: 

- **Randomness**: the goal of the program is to find a **minimum connected dominating set** of a graph. There's no deterministic algorithm found that can solve this question in polynomial time. Also, simply use MST or SPT with greedy leaf pruning does not generate a very good solution because

  - There are too many ways to prune the tree
  - Pruning a leaf may increase the cost. 
  - Minimum connected dominating set is not necessarily a subset of some MST or SPT. 

  Therefore, we rely on randomness because the there's no need for a program to track which solution has been already explored so the memory cost is relatively cheap. Also, near-best solution can be found with a high probability. 

- **Solution Quality: ** For each iteration we have multiple "workers". The probability of finding a better solution is much higher than only focusing on one working solution in an iteration. Also, the answer can only get better and better. 

## Other approaches we tried

- **Greedy leaf pruning:** In this algorithm we choose a random SPT or MST in the graph and keeps pruning leaves until it is impossible to prune more leaves. This algorithm has around 30% higher cost because minimum connected dominating set may not be in any MST or SPT.
- **Naïve worker**: In this algorithm, we only has one `worker` at a time. Keep call `work` on the worker until the worker cannot find a better answer and repeat. This algorithm has 15% higher cost compared with the algorithm we used (giving some computation power and computational time). 

## Computational Resources

We use instructional machines for computation. Here are some details: 

```
Server: hive13
Address: hive13.cs.berkeley.edu
Running Time: ~144 hours
Process name: ./abc_client.py --parallel
```

Here's the `htop` information related to the computation task: 

![image-20200430170630919](assets/image-20200430170630919.png)