from typing import Union
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
        else:
            self.bn = net

    # TODO: Remove TODO's as we go
    # Hit it, hit it, get it, get it

    # TODO Given three sets of variables X, Y , and Z, determine whether X
    # is independent of Y given Z. (5pts)
    def d_separation(self, X, Z, Y):
        pass

    # TODO Given a set of variables X in the Bayesian network,
    # compute a good ordering for elimination of X based on the min-degree
    # heuristics (2pts) and the min-fill heuristics (2pts).
    # (Hint: you get the interaction graph ”for free” from the BayesNet class)
    def ordering(self):
        pass

    # TODO Given a set of query variables Q and evidence E, node- and
    # edge-prune the Bayesian network s.t. queries of the form P(Q|E)
    # can still be correctly calculated (5pts).
    def network_pruning(self):
        self.node_prune()
        self.edge_prune()

    # TODO Node pruning
    def node_prune(self):
        pass
    
    # Edge pruning
    def edge_prune(self):
        pass

    # TODO Given query variables Q and a possibly empty evidence E, compute
    # the marginal distribution P(Q|E) (12pts). (Note that Q is a subset of
    # the variables in the Bayesian network X with Q⊂X but can also be Q=X.)
    def marginal_distributions(self):
        pass

    # TODO MAP and MEP: Given a possibly empty set of query variables Q and an
    # evidence E, compute the most likely instantiations of Q (12pts).
    def MAP(self):
        pass

    # TODO liberate the settlements
    def MEP(self):
        pass

    # Mainly for trying things
    def main():
        pass

if __name__ == "__main__":
    net_path = "testing/dog_problem.BIFXML"
    reasoner = BNReasoner(net=net_path)
    reasoner.bn.draw_structure()

