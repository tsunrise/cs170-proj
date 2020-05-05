# CS 170 Project Spring 2020

The goal of this project is to find a connected dominating set that minimizes average pairwise routing distance. Check the <a href = "https://github.com/tsunrise/cs170-proj/blob/master/spec.pdf">spec</a> for details. 

Files:

- `abc_client.py`: main function to run
- `parse.py`: functions to read/write inputs and outputs
- `solver.py`: where you should be writing your code to solve inputs
- `utils.py`: contains functions to compute cost and validate NetworkX graphs

**How to run:** 

- Please put your input files inside `./inputs/`. 
- To run the solver on **all** inputs, enter: 
```shell
./abc_client.py -a
```

- To run the solver in parallel on **all** inputs, enter:

```shell
./abc_client.py --parallel -a
```

If you are not using bash, remember the add `python3` at beginning, i.e.

```shell
python3 ./abc_client.py -a
python3 ./abc_client.py --parallel -a
```

- If you only want to run inputs whose previous output was not first rank on the leaderboard: 

  - Go to the leaderboard website: https://berkeley-cs170.github.io/project-leaderboard/

  - Install the following user script written by myself: https://greasyfork.org/zh-CN/scripts/402559-cs170-leaderboard-data-download (Make sure to change your team name in `abc_client.py`)

  - On the leaderboard site, on right top, click "Request Download", then wait for a second until the button "Get JSON" appears. Download the JSON. 

  - Put `rank.json` in the same directory of the code. 

  - Run one of the following code: 

    ```shell
    ./abc_client.py
    ./abc_client.py --parallel # if you want to run the code in parallel. 
    ```

    
