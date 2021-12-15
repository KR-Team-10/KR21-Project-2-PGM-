from typing import Union, List
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format
        or BayesNet object
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
    # TODO We need to check my implementation because in the slides its XYZ insted of queries and evidence
    def network_pruning(self, X: List[str], Y: List[str], Z: List[str]):
        """
        Prunes the edges and nodes of the structure.
        Deletes every leaf node W ∉ X∪Y∪Z.
        Deletes all edges outgoing from nodes in Z.
        :param X: A list of variable names.
        :param Y: A list of variable names.
        :param Z: A list of variable names. X and Y are d-separated with respect to Z.
        """
        self.__node_prune(X, Y, Z)
        self.__edge_prune(X, Y, Z)

    # TODO Please check my work
    # Deletes every leaf node W ∉ X∪Y∪Z
    def __node_prune(self, X: List[str], Y: List[str], Z: List[str]):
        prune = []
        for W in self.bn.get_all_variables():
            if self.bn.descendants(W) == []:  # If W is a leaf node
                if W not in (X + Y + Z):  # If W ∉ X∪Y∪Z
                    prune.append(W)
        for p in prune:
            self.bn.del_var(p)

    # TODO please check my work
    # Deletes all edges outgoing from nodes in Z
    def __edge_prune(self, X: List[str], Y: List[str], Z: List[str]):
        prune = []
        for edge in self.bn.structure.edges:
            if edge[0] in Z:
                prune.append(edge)
        for p in prune:
            self.bn.del_edge(p)

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
    # net_path = "testing/dog_problem.BIFXML"
    net_path = "testing/dog_problem.BIFXML"

    reasoner = BNReasoner(net=net_path)

    reasoner.bn.draw_structure()
    reasoner.network_pruning(1, 1, ["dog-out"])
    reasoner.bn.draw_structure()

    # print(reasoner.bn.parents("dog-out"))
    # print(reasoner.bn.descendants("dog-out"))
    # print(reasoner.bn.non_descendants("dog-out"))

    # reasoner.bn.draw_structure()


if __name__ == "__main__":
    main()
