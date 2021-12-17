from typing import Union, List, Tuple, Dict
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
    def network_pruning(self, Q: List[str], E: Dict[str, bool]):
        """
        Prunes the edges and nodes of the structure.
        Deletes every leaf node W ∉ Q∪E.
        Deletes all edges outgoing from nodes in E.
        :param Q: A list of variable names.
        :param E: A dictionary mapping variable names to their indicated truth assignments.
        """
        self.__node_prune(Q, E)
        self.__edge_prune(E)

    # TODO Please check my work
    # Deletes every leaf node W ∉ Q∪E
    def __node_prune(self, Q: List[str], E: Dict[str, bool]):
        prune = []
        for W in self.bn.get_all_variables():
            if self.bn.descendants(W) == []:  # If W is a leaf node
                if W not in (Q + list(E)):  # If W ∉ Q∪E
                    prune.append(W)
        [self.bn.del_var(p) for p in prune]

    # Deletes all edges U from nodes in E
    def __edge_prune(self, E: Dict[str, bool]):
        prune = []
        for edge in self.bn.structure.edges:
            if edge[0] in list(E):
                prune.append(edge)
        for p in prune:
            self.__del_edge_and_replace_cpt(edge=p, evidence=(p[0], E[p[0]]))

    def __del_edge_and_replace_cpt(self, edge: Tuple[str, str], evidence: Tuple[str, bool]):
        self.bn.del_edge(edge)
        # We neeed to implement CPT replacement for pruned edges
        self.__replace_cpt(cpt_variable=edge[1], evidence=evidence)

    # TODO Look at slide 9 of PGM-4
    def __replace_cpt(self, cpt_variable: str, evidence: Tuple[str, bool]):
        evidence_var, truth = evidence

        old_cpt = self.bn.get_cpt(cpt_variable)
        new_cpt = old_cpt[old_cpt[evidence_var] == truth].drop([evidence_var], axis=1, inplace=False)
        self.bn.update_cpt(variable=cpt_variable, cpt=new_cpt)



    # TODO Given query variables Q and a possibly empty evidence E, compute
    # the marginal distribution P(Q|E) (12pts). (Note that Q is a subset of
    # the variables in the Bayesian network X with Q⊂X but can also be Q=X.)
    # def marginal_distribution(self, Q: list[str], E: Dict[str, bool]):
    #     pass

    # TODO MAP and MEP: Given a possibly empty set of query variables Q and an
    # evidence E, compute the most likely instantiations of Q (12pts).
    def MAP(self):
        pass

    # TODO liberate the settlements
    def MEP(self):
        pass


# Mainly for trying things
def main():

    # variables = list("ABCDEF")
    # edges = [
    #     ("A", "B"),
    #     ("A", "C"),
    #     ("B", "D"),
    #     ("B", "E"),
    #     ("C", "E"),
    #     ("C", "F"),
    # ]
    # bn = BayesNet()
    # bn.create_bn(variables, edges, {x: None for x in variables})
    # reasoner = BNReasoner(bn)
    # reasoner.network_pruning(Q=[], E={"A": True, "C": False})
    # reasoner.bn.draw_structure()

    net_path = "testing/dog_problem.BIFXML"

    reasoner = BNReasoner(net=net_path)
    reasoner.network_pruning(Q=[], E={"family-out": True})

    reasoner.bn.draw_structure()


if __name__ == "__main__":
    main()
