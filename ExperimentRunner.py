from BNReasoner import BNReasoner
import random
import os

RUNS = 10


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

    def experiment(self, reasoner: BNReasoner, filename, results_file):
        # Get the number of variables in the network
        n_var = len(reasoner.bn.get_all_variables())
        # The size of the MAP and MPE queries will be a ratio of the number of variables
        query_size = round(n_var / 3)

        for heuristic in ["fill", "degree", "rand"]:
            pass


def main():
    runner = ExperimentRunner(bn_folders=["bayes"])
    runner.run()


if __name__ == "__main__":
    main()
