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
                "n_nodes,minFillMPE,minDegMPE,minFillMAP,minDegMAP,randMPE,randMAP\n"
            )

        # For each dataset of different size
        for folder in self.bn_folders:
            # For each Bayesian Network within that dataset
            for filename in os.listdir(os.path.join(self.data_directory, folder)):
                # Create a BNReasoner with that Bayesian Network
                net_path = os.path.join(self.data_directory, folder, filename)

                reasoner = BNReasoner(net_path)

                self.experiment(
                    reasoner, filename, results_file=f"results/results_{i}.csv"
                )

    def get_query_and_evidence(self, bn: BayesNet):
        variable_names = bn.get_all_variables()
        n_var = len(variable_names)

        Q_n = n_var // QUERY_RATIO
        E_n = Q_n // 3

        Q = random.sample(variable_names, Q_n)

        E_vars = Q[:E_n]

        E = {v: random.choice([True, False]) for v in E_vars}

        return Q, Q[E_n:], E

    def experiment(self, reasoner: BNReasoner, filename, results_file):
        # Get the number of variables in the network
        n_var = len(reasoner.bn.get_all_variables())
        # The size of the MAP and MPE queries will be a ratio of the number of variables
        query_size = round(n_var / 3)

        Q, Q_E, E = self.get_query_and_evidence(reasoner.bn)
        print(len(Q), len(Q_E), len(E))

        for heuristic in ["fill", "degree", "rand"]:
            # TODO Time minFillMPE
            fill_mpe_start = time()
            fill_mpe = time() - fill_mpe_start

            # TODO Time minDegMPE
            deg_mpe_start = time()
            deg_mpe = time() - deg_mpe_start

            # TODO Time minFillMAP
            fill_map_start = time()
            fill_map = time() - fill_map_start

            # TODO Time minDegMAP
            deg_map_start = time()
            deg_map = time() - deg_map_start

            # TODO Time randMPE
            rand_mpe_start = time()
            rand_mpe = time() - rand_mpe_start

            # TODO Time randMPE
            rand_map_start = time()
            rand_map = time() - rand_map_start

            with open(results_file, "a") as datafile:
                datafile.write(
                    f"{n_var},{fill_mpe},{deg_mpe},{fill_map},{deg_map},{rand_mpe},{rand_map}\n"
                )


def main():
    runner = ExperimentRunner(bn_folders=["bayes"])
    runner.run()


if __name__ == "__main__":
    main()
