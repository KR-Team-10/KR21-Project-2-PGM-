from BNReasoner import BNReasoner
from BayesNet import BayesNet
import random
import os
from time import time

RUNS = 10
QUERY_RATIO = 3


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

                net_path = (
                    "/Users/nedim.azar/Desktop/VU/KR21-Project-2-PGM-/bayes/40.xml"
                )
                reasoner = BNReasoner(net_path)

                self.experiment(
                    reasoner, filename, results_file=f"results/results_{i}.csv"
                )

    def get_query_and_evidence(self, bn: BayesNet):
        variable_names = bn.get_all_variables()
        n_var = len(variable_names)

        Q_n = n_var // 2
        E_n = Q_n // 2

        Q = random.sample(variable_names, Q_n)

        E_vars = Q[:E_n]

        E = {v: random.choice([True, False]) for v in E_vars}

        return Q, Q[E_n:], E

    def experiment(self, reasoner: BNReasoner, filename, results_file):
        # Get the number of variables in the network
        n_var = len(reasoner.bn.get_all_variables())
        # The size of the MAP and MPE queries will be a ratio of the number of variables
        query_size = round(n_var / 3)

        for x in range(10):
            print(f"RUN #{x}")
            Q, Q_E, E = self.get_query_and_evidence(reasoner.bn)

            reasoner.network_pruning(Q=Q_E, E=E)

            pi_deg = reasoner.ordering(heuristic="degree")
            pi_fill = reasoner.ordering(heuristic="fill")
            pi_rand = reasoner.ordering(heuristic="rand")

            print(reasoner.marginal_distribution(Q=Q, E={}, pi=pi))
            # "n_nodes,minDegMAP,minFillMAP,randMAP,minDegMPE,minFillMPE,randMPE\n"

            try:
                # Time minDegMAP
                deg_map_start = time()
                reasoner.MAP(Q=Q_E, E=E, pi=pi_deg)
                deg_map = time() - deg_map_start
                print("MAP with DEGREE heuristic:", deg_map)
            except Exception:
                print("MAP with DEGREE failed :(")
                deg_map = None

            try:
                # Time minFillMAP
                fill_map_start = time()
                reasoner.MAP(Q=Q_E, E=E, pi=pi_fill)
                fill_map = time() - fill_map_start
                print("MAP with FILL heuristic:", fill_map)
            except Exception:
                print("MAP wih FILL failed :(")

            try:
                # Time randMAP
                rand_map_start = time()
                reasoner.MAP(Q=Q_E, E=E, pi=pi_rand)
                rand_map = time() - rand_map_start
                print("MAP with RAND heuristic:", rand_map)
            except Exception:
                print("MAP wih RAND failed :(")

            try:
                # Time minDegMPE
                deg_mpe_start = time()
                reasoner.MPE(E=E, pi=pi_deg)
                deg_mpe = time() - deg_mpe_start
                print("MPE with DEGREE heuristic:", deg_mpe)
            except:
                print("MPE with DEGREE failed :(")

            try:
                # Time minFillMPE
                fill_mpe_start = time()
                reasoner.MPE(E=E, pi=pi_fill)
                fill_mpe = time() - fill_mpe_start
                print("MPE with FILL heuristic:", fill_mpe)
            except Exception:
                print("MPE with FILL failed :(")
            try:
                # Time randMPE
                rand_mpe_start = time()
                rand_mpe = time() - rand_mpe_start
                print("MPE with RAND heuristic:", rand_mpe)
            except:
                print("MPE with RAND failed :(")

            with open(results_file, "a") as datafile:
                datafile.write(
                    f"{n_var},{deg_map},{fill_map},{rand_map},{deg_mpe},{fill_mpe},{rand_mpe}\n"
                )


def main():
    runner = ExperimentRunner(bn_folders=["bayes"])
    runner.run()


if __name__ == "__main__":
    main()
