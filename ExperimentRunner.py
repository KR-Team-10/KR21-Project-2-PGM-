from BNReasoner import BNReasoner
from BayesNet import BayesNet
import random
import os
from time import time
from copy import deepcopy

N_RUNS = 100  # The number of random queries to run on each BN.
N_QUERY = 3  # Change this to set the number of Query variables, a third by default.
N_EVIDENCE = 2  # Change this to set the number of Evidence variables, half of the Query variables by default.


class ExperimentRunner:
    def __init__(self, data_directory="", bn_folders=["large"]):
        self.data_directory = data_directory
        self.bn_folders = bn_folders

    def run(self):
        i = 0
        while os.path.exists(f"results/results_{i}.csv"):
            i += 1
        with open(f"results/results_{i}.csv", "w") as results:
            results.write(
                "n_nodes,minDegMAP,minFillMAP,randMAP,minDegMPE,minFillMPE,randMPE\n"
            )

        # For each dataset of different size
        for folder in self.bn_folders:
            # For each Bayesian Network within that dataset
            for filename in os.listdir(os.path.join(self.data_directory, folder)):
                # Create a BNReasoner with that Bayesian Network
                net_path = os.path.join(self.data_directory, folder, filename)

                # reasoner = BNReasoner(net_path)

                self.experiment(net_path, results_file=f"results/results_{i}.csv")

    def get_query_and_evidence(self, bn: BayesNet):
        Q = []
        E = {}
        while Q == [] or E == {}:
            variable_names = bn.get_all_variables()
            n_var = len(variable_names)

            Q_n = n_var // N_QUERY
            E_n = Q_n // N_EVIDENCE

            Q = random.sample(variable_names, Q_n)

            E_vars = Q[:E_n]

            E = {v: random.choice([True, False]) for v in E_vars}

        return Q, Q[E_n:], E

    def query_evidence(self, bn):
        variables = deepcopy(bn.get_all_variables())

        n = len(variables) // 5

        Q = []
        E = {}

        for x in range(n):
            Q.append(variables.pop(random.randint(0, n - 1)))
            E[variables.pop(random.randint(0, n - 1))] = random.choice([True, False])
        return Q, E

    def experiment(self, filename, results_file):
        n_var = None
        for x in range(N_RUNS):
            print(f"RUN #{x}")

            reasoner = BNReasoner(filename)
            n_var = len(reasoner.bn.get_all_variables())

            Q_E, E = self.query_evidence(reasoner.bn)
            reasoner.network_pruning(Q=Q_E, E=E)

            pi_deg = reasoner.ordering(heuristic="degree")
            pi_fill = reasoner.ordering(heuristic="fill")
            pi_rand = reasoner.ordering(heuristic="rand")

            try:
                # Time minDegMAP
                deg_map_start = time()
                reasoner.MAP(Q=Q_E, E=E, pi=pi_deg)
                deg_map = time() - deg_map_start
                print("MAP with DEGREE heuristic:", deg_map)
            except Exception:
                print("MAP with DEGREE failed :(")
                deg_map = None
                print(Q_E, E, sep="\n")

            try:
                # Time minFillMAP
                fill_map_start = time()
                reasoner.MAP(Q=Q_E, E=E, pi=pi_fill)
                fill_map = time() - fill_map_start
                print("MAP with FILL heuristic:", fill_map)
            except Exception:
                print("MAP wih FILL failed :(")
                fill_map = None

            try:
                # Time randMAP
                rand_map_start = time()
                reasoner.MAP(Q=Q_E, E=E, pi=pi_rand)
                rand_map = time() - rand_map_start
                print("MAP with RAND heuristic:", rand_map)
            except Exception:
                print("MAP wih RAND failed :(")
                rand_map = None

            try:
                # Time minDegMPE
                deg_mpe_start = time()
                reasoner.MPE(E=E, pi=pi_deg)
                deg_mpe = time() - deg_mpe_start
                print("MPE with DEGREE heuristic:", deg_mpe)
            except:
                print("MPE with DEGREE failed :(")
                deg_mpe = None

            try:
                # Time minFillMPE
                fill_mpe_start = time()
                reasoner.MPE(E=E, pi=pi_fill)
                fill_mpe = time() - fill_mpe_start
                print("MPE with FILL heuristic:", fill_mpe)
            except Exception:
                print("MPE with FILL failed :(")
                fill_mpe = None

            try:
                # Time randMPE
                rand_mpe_start = time()
                reasoner.MPE(E=E, pi=pi_rand)
                rand_mpe = time() - rand_mpe_start
                print("MPE with RAND heuristic:", rand_mpe)
            except:
                print("MPE with RAND failed :(")
                rand_mpe = None

            with open(results_file, "a") as datafile:
                datafile.write(
                    f"{n_var},{deg_map},{fill_map},{rand_map},{deg_mpe},{fill_mpe},{rand_mpe}\n"
                )


def main():
    runner = ExperimentRunner(bn_folders=["bayes"])
    runner.run()


if __name__ == "__main__":
    main()
