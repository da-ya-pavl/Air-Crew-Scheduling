Aim:

You are required to implement simulated annealing (SA), a standard Binary Genetic Algorithm (BGA) and an improved BGA to solve three airline crew scheduling problems (discussed in Week4_Tutorial.pdf). The SA and standard BGA implementations should be straightforward. For the improved BGA, we will follow [1] to implement several specific operators, including initialisation, mutation, and refinement. We will also implement the stochastic ranking method [2] for handling constraints (as introduced in week 4's lectures). The problems we will solve are sppnw41, sppnw42 and sppnw43 in OR-Library:

sppnw41.txt 

sppnw42.txt

sppnw43.txt

The format of these files is explained at Set partitioning.html

Note: Although the default parameter values of each algorithm should work well, you should still check the results. If the results are too far from the optimal results, you may need to tune the parameters to achieve good-enough results. After tuning, the parameter values for each algorithm should remain the same across all three problems.
Requirements:

    You must implement the three algorithms from scratch. You can use any programming language. However, if you want to use languages other than MATLAB or Octave, you should make your program executable. For example, if you use Java, you need to compile it. If you use Python, ensure it can be run in a Python online IDE.   
    Correct implementations of the SA and Standard BGA algorithms. We expect your implementations to generate reasonably good results, i.e., close to the optimal results reported in [1].  (8 points)
    For the improved BGA,  
        Implement the pseudo-random initialisation method (Algorithm 2, p341 in [1]) (2 points).  Note: This algorithm is similar to the stochastic local search algorithm for the set cover problem. 
        Implement the stochastic ranking method described in [2] for constraint handling (2 points). 
        Implement the heuristic improvement operator (Algorithm 1, p331 in [1]) (3 points).
        Again, we will assign marks based on the performance of your implementation, as we expect that the correct implementation should produce results close to the optimal values reported in [1].
    Write a report to:
        Introduce the SA, the standard BGA and the improved BGA. You need to use a flowchart and pseudo-code to explain the three algorithms. (4 points)
        For each benchmark problem, list the average result and standard deviation obtained over 30 independent runs of each algorithm (1 point)
        Compare the results from SA, the standard BGA and the improved BGA with an in-depth discussion.  (4 points)
        Discuss the similarity and difference between the ranking replacement method in [1] and the stochastic ranking method in [2] (1 point)
    Please submit your report with your source code in a zip file. 

Reference:

[1] P.C. Chu, J.E. Beasley, Constraint Handling in Genetic Algorithms: The Set Partitioning Problem, Journal of Heuristics, 11: 323–357 (1998).pdf

[2] T.P. Runarsson, X. Yao Stochastic ranking for constrained evolutionary optimization, IEEE Trans. on Evolutionary Computation, 4(3): 284-294 (2000).pdf