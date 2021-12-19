from BNReasoner import BNReasoner
import random
import os

RUNS = 10


class ExperimentRunner:
    def __init__(self, data_directory="", bn_folders=["large"]):
        self.data_directory = data_directory
        self.bn_folders = bn_folders

    def run(self):
        # For each dataset of different size
        for folder in self.bn_folders:
            # For each Bayesian Network within that dataset
            for filename in os.listdir(os.path.join(self.data_directory, folder)):
                # Create a BNReasoner with that Bayesian Network
                net_path = os.path.join(self.data_directory, folder, filename)

                try:
                    reasoner = BNReasoner(net=net_path)
                    print(filename, len(reasoner.bn.get_all_variables()))
                except Exception:
                    os.remove(net_path)
                    print("Problem with file:", filename)


def main():
    runner = ExperimentRunner(bn_folders=["bayes"])
    runner.run()


if __name__ == "__main__":
    main()
