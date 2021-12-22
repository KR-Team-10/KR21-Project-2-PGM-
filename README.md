# README useful information regarding the project files and functions

# Test files 

+ All the networks used to test the implemented functionalitites are in the folder "/testing".
+ + The network designed for our use-case is in the file 'psyc_disorders.BIFXML' 

+ All the networks for the experiment for Task 2 are in the folder /bayes


# About running test cases
+ All the test cases for the Task 1  and Task 3 are contained in the file 'BNReasoner.py' along with the implementation of the functionalities.
+ All answers for the questions regarding the use-case were runned in the use_case_questions() function of the 'BNReasoner.py" file.

# About running the experiment
+ The experiment for Task 2 contained in the file 'ExperimentRunner.py' and can be done again by running the file
+ 

# Functionalities implemented
+ Next, a list of all functionalities implemented for Task 1


+ BNReasoner.d_separation(self, X: List[str], Z: List[str], Y: List[str]):

    Returns True when X is d-separated from Y given Z


+ BnReasoner.ordering(heuristic="degree"):    

    @param heuristic = "degree" | "fill" | "rand"
            
    Returns an ordering pi given one of the three heuristics


+ BnReasoner.network_pruning(self, Q: List[str], E: Dict[str, bool])

    Node and Edge prunning of the network given a query Q and Evidence E


+ BnReasoner.marginal_distribution(self, Q: List[str], E: Dict[str, bool], pi: List[str])
    
    Returns the marginal probability distribution of the network for a query Q and evidence E, given an ordering for the variable elimination pi.


+ BnReasoner.def MAP(self,Q: List[str],E: Dict[str, bool], pi: List[str])

    Returns the MAP of the network for the variables Q, evidence E and given and ordering pi

+ BnReasoner.MPE(self, E: Dict[str, bool], pi: List[str]):    
    Returns the MPE of the network variavles given some evidence E and ordering pi



