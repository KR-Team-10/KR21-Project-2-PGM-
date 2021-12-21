from math import factorial
from typing import Union, List, Tuple, Dict

from numpy import empty, multiply
from BayesNet import BayesNet
from itertools import combinations,product
from copy import deepcopy
import pandas as pd


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet], XML=True):
        """
        :param net: either file path of the bayesian network in BIFXML format
        or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net, XML=XML)
        else:
            self.bn = net
        self.dsep_bn = deepcopy(self.bn)

    # TODO: Remove TODO's as we go
    # Hit it, hit it, get it, get it

    # d-Separation (5pts)
    def d_separation(self, X: List[str], Z: List[str], Y: List[str]) -> bool:
        """
        Given disjoint sets of variables X, Y and Z, determines whether X is independent
        of Y given Z.

        :param X: A list of variable name strings.
        :param Z: A list of variable name strings.
        :param Y: A list of variable name strings.
        :return: True if X s is independent of Y given Z, False otherwise.
        """
        nodes_pruned, edges_pruned = True, True
        while nodes_pruned or edges_pruned:
            if nodes_pruned:
                nodes_pruned = self.__dsep_node_prune(X, Y, Z)
            if edges_pruned:
                edges_pruned = self.__dsep_edge_prune(X, Y, Z)
        return self.dsep_bn.disconnected(X, Y)

    # Ordering (2 + 2pts)
    def ordering(self, heuristic="degree"):
        """
        Given a set of variables X in the Bayesian network,
        computes a good ordering for elimination of X based on the min-degree or min-fill heuristic.

        :param heuristic: Set to 'degree' for min-degree ordering or 'fill' for min-fill ordering.
        """
        if heuristic.lower() not in ["degree", "fill"]:
            raise Exception
        if heuristic == "degree":
            return self.__min_degree_order()
        return self.__min_fill_order()

    # Network Pruning (5pts)
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

    def __get_compatible_cpts(self,E: Dict[str, bool]):
        S = []
        cpts = {}

        self.__node_prune(Q, E)
        self.__edge_prune(E)

        for var in self.bn.get_all_variables():
            cpts[var] = self.bn.get_cpt(var)
            if len(E) != 0:
                S.append(
                    self.bn.get_compatible_instantiations_table(pd.Series(E), cpts[var])
                )
            else:
                S.append(cpts[var])

        return S

    # TODO Given query variables Q and a possibly empty evidence E, compute
    # the marginal distribution P(Q|E) (12pts). (Note that Q is a subset of
    # the variables in the Bayesian network X with Q⊂X but can also be Q=X.)
    # Marginal Distributions (12pts)
    # TODO ve_pr2 algorithm
    def marginal_distribution(self, Q: List[str], E: Dict[str, bool], pi: List[str]):
        print("Q = ", Q)
        print("E= ", E)
        print("pi = ", pi)

        S = self.__get_compatible_cpts(E)

        query_joint_prob = self.joint_distribution(Q,S,pi)

        posterior_marginal_distribution = self.normalize(query_joint_prob)

        return posterior_marginal_distribution


    # TODO MAP and MEP: Given a possibly empty set of query variables Q and an
    # evidence E, compute the most likely instantiations of Q (12pts).
    def MAP(self,E: Dict[str, bool], pi: List[str]):
        print("oli")
        

    # TODO liberate the settlements
    def MPE(self,E: Dict[str, bool], pi: List[str] ):
        Q = self.bn.get_all_variables()
        mpe_instantations = {}

        self.__node_prune(Q,E)
        self.__edge_prune(E)

        S = self.__get_compatible_cpts(E)
        [print(S[i]) for i in range (0,len(S))]

        for i in range(0, len(pi)):
            
            pi_i = pi[i]
            print("\nPI({}) = : {}".format(i,pi_i))

            #get factors mentioning pi(i)
            factors_including_var = self.__get_factors_including_var(S,pi_i)
            
            # multiply all factors mentioning variable pi(i)
            f = self.multiply_factors(factors_including_var,pi_i)
            
            f_i = self.max_out_var(f,pi_i)
            
            #remove elements factors_including_var from S 
            for factor in factors_including_var:
                arr = [factor.sort_index().sort_index(axis=1).equals(s_factor.sort_index().sort_index(axis=1))  for s_factor in S]
                for j in range(0,len(arr)):
                    if arr[j] == True: S.pop(j) 
                
            #then add new factor f_i to S
            S.append(f_i)
                
            print("\n-result S: \n")
            [print(S[i]) for i in range(0,len(S))]                
            print("_____________________________________________")

            # for s in S:
            #     print(s)            
            #     if(len(s.index) == 1):
            #         print("LENGHT 1")
            #         # print(s.index)
            #         print(s.iloc[0])
            #         print(s.iloc[0][0])
            #         # mpe_instantations[] = s.iloc[0][0]
            # print("_____________________________________________")

        print(mpe_instantations)
        # S = self.multiply_factors(S,'')        

                
    def joint_distribution(self, Q: List[str], S: List[pd.DataFrame], pi: List[str]):

        # [print(S[i]) for i in range (0,len(S))]
        for i in range(0, len(pi)):

            pi_i = pi[i]
            # print("\nPI({}) = : {}".format(i, pi_i))

            # get factors mentioning pi(i)
            factors_including_var = self.__get_factors_including_var(S, pi_i)

            # multiply all factors mentioning variable pi(i)
            print(factors_including_var)
            f = self.multiply_factors(factors_including_var, pi_i)

            # sum out
            if pi_i not in Q:
                f_i = self.sum_out_var(f, pi_i)
            else:
                f_i = f

            # remove elements factors_including_var from S
            for factor in factors_including_var:
                arr = [
                    factor.sort_index()
                    .sort_index(axis=1)
                    .equals(s_factor.sort_index().sort_index(axis=1))
                    for s_factor in S
                ]
                for i in range(0, len(arr)):
                    if arr[i] == True:
                        S.pop(i)

            # then add new factor f_i to S
            S.append(f_i)

            # print("\n-result S: \n")
            # [print(S[i]) for i in range(0, len(S))]
            # print("_____________________________________________")
        print("S:", S)
        S = self.multiply_factors(S, "")

        return S

    def multiply_factors(self, factors: List[pd.DataFrame], var: str) -> pd.DataFrame:
        print((factors))
        if len(factors) == 1:
            return factors[0]
        else:
            while len(factors) > 1:
                f1 = factors[0]
                f2 = factors[1]

                if not var:
                    var = list(f1.columns)
                    var.remove("p")
                    print(var)
                else:
                    var = [var] if isinstance(var, str) else var[0]

                mult = f1.merge(f2, on=var)
                mult["p"] = mult.p_x * mult.p_y
                mult = mult.drop(["p_x", "p_y"], axis=1)

                factors = factors[2:]
                factors.append(mult)
        print(factors)
        return factors[0]

    def sum_out_var(self, factor: pd.DataFrame, var: str) -> pd.DataFrame:

        variables = list(factor.columns)
        variables.remove("p")
        variables.remove(var)

        if(len(variables)>0):
            factor = factor.groupby(variables, as_index=False).agg('sum')
            
            factor = factor.drop([var],axis=1)
        
       
        return factor

    def max_out_var(self, factor: pd.DataFrame, var: str) -> pd.DataFrame:
        # print("**************************************************************")
        # print("MAX OUT")
        print("factor: \n",factor)
        variables = list(factor.columns)
        variables.remove("p")
        variables.remove(var)
        print("group by variables: \n",variables)

        if(len(variables)>0):
            
            agg_dict =  {
                        'p':lambda x : max(x),    # Sum duration per group

                    
                    }

            factor = factor.groupby(
                variables
                ,as_index=False
                ).agg(
                    agg_dict
                )                
                        
            
            return factor        

        return factor #else
        

    def normalize(self, joint_probability: pd.DataFrame):

        normalize_factor = joint_probability["p"].sum()
        joint_probability.p = joint_probability.p / normalize_factor

        return joint_probability

    def __get_factors_including_var(self, factors: List[pd.DataFrame], k: str):
        """
         Get the subset of factors mentioning variable k

        :param factors: set of factors
        :param k: variable k mentioned in some of the factors
        """
        V = set()
        factors_k = []

        for factor in factors:
            if k in list(factor.columns):
                factors_k.append(factor)

        return factors_k

    # Get's the set of all variables of a list of factors
    # Or the set of variables that the resulting factor will be over
    def __get_vars(self, factors: List[pd.DataFrame]):
        V = set()
        for factor in factors:
            for variable in list(factor.columns):
                if variable != "p":
                    V.add(variable)
        return V

    def dsep_network_pruning(self, X: List[str], Y: List[str], Z: List[str]):
        """
        Prunes the edges and nodes of the structure.
        Deletes every leaf node W ∉ X∪Y∪Z.
        Deletes all edges outgoing from nodes in Z.

        :param X: A list of variable name strings.
        :param Y: A list of variable name strings.
        :param Z: A list of variable name strings. X and Y are d-separated with respect to Z.
        """
        self.__dsep_node_prune(X, Y, Z)
        self.__dsep_edge_prune(X, Y, Z)

    # Deletes every leaf node W ∉ X∪Y∪Z
    def __dsep_node_prune(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        prune = []
        for W in self.dsep_bn.get_all_variables():
            if self.dsep_bn.descendants(W) == []:  # If W is a leaf node
                if W not in (X + Y + Z):  # If W ∉ X∪Y∪Z
                    prune.append(W)
        for p in prune:
            self.dsep_bn.del_var(p)
        return prune != []

    # Deletes all edges outgoing from nodes in Z
    def __dsep_edge_prune(self, X: List[str], Y: List[str], Z: List[str]) -> bool:
        prune = []
        for edge in self.dsep_bn.structure.edges:
            if edge[0] in Z:
                prune.append(edge)
        for p in prune:
            self.dsep_bn.del_edge(p)
        return prune != []

    # Deletes every leaf node W ∉ Q∪E
    def __node_prune(self, Q: List[str], E: Dict[str, bool]):
        prune = []
        for W in self.bn.get_all_variables():
            if self.bn.descendants(W) == []:  # If W is a leaf node
                if W not in (Q + list(E)):  # If W ∉ Q∪E
                    prune.append(W)
        [self.bn.del_var(p) for p in prune]

    # Deletes all edges U ourgoing from nodes in E
    def __edge_prune(self, E: Dict[str, bool]):
        prune = []
        for edge in self.bn.structure.edges:
            if edge[0] in list(E):
                prune.append(edge)
        for p in prune:
            self.__del_edge_and_replace_cpt(edge=p, evidence=(p[0], E[p[0]]))

    def __del_edge_and_replace_cpt(
        self, edge: Tuple[str, str], evidence: Tuple[str, bool]
    ):
        self.bn.del_edge(edge)
        self.__replace_cpt(cpt_variable=edge[1], evidence=evidence)

    def __replace_cpt(self, cpt_variable: str, evidence: Tuple[str, bool]):
        evidence_var, truth = evidence

        old_cpt = self.bn.get_cpt(cpt_variable)
        new_cpt = old_cpt[old_cpt[evidence_var] == truth].drop(
            [evidence_var], axis=1, inplace=False
        )
        self.bn.update_cpt(variable=cpt_variable, cpt=new_cpt)

    # Get the MINIMAL DEGREE order of variable eliminiation
    def __min_degree_order(self):
        interaction = self.bn.get_interaction_graph()
        pi = []
        for i in range(0, len(interaction.nodes)):
            # get node with min degree
            degrees = {}
            for node in interaction.nodes:
                degrees[node] = 0
                for a, b in interaction.edges:
                    if a == node or b == node:
                        degrees[node] += 1

            min_degree_node = min(degrees, key=degrees.get)
            pi.append(min_degree_node)

            # get neighbors of min grade node
            neighbors = []
            for a, b in interaction.edges:
                if a == min_degree_node:
                    neighbors.append(b)
                elif b == min_degree_node:
                    neighbors.append(a)

            # add an edge between every pair of non-adjacent neighbors of min_degree_node
            if len(neighbors) > 1:
                for comb in list(combinations(neighbors, 2)):
                    interaction.add_edge(
                        comb[0], comb[1]
                    )  # if the edge already exists, then this don't do anything

            # delete variable min_degree_node
            interaction.remove_node(min_degree_node)

        return pi

    # Get the MINIMAL FILL order of variable eliminiation
    def __min_fill_order(self):
        interaction = self.bn.get_interaction_graph()
        pi = []

        for i in range(0, len(interaction.nodes)):
            # print("nodescls = ",interaction.nodes)
            counter = {}
            add_edges = {}
            for node in interaction.nodes:

                add_edges[node] = []
                neighbors = []
                for a, b in interaction.edges:
                    if a == node:
                        neighbors.append(b)
                    elif b == node:
                        neighbors.append(a)

                # count how many edges needs to be created if this variable is removed
                counter[node] = 0
                if len(neighbors) > 1:
                    for comb in list(combinations(neighbors, 2)):
                        if comb not in interaction.edges or tuple(
                            reversed(comb)
                        ) not in (interaction.edges):
                            counter[node] += 1
                            add_edges[node].append(comb)

            # get the node which add the min number of edges if it is removed
            min_fill_node = min(counter, key=counter.get)

            # add an edge between every pair of non-adjacent neighbors of min_degree_node
            if len(add_edges[min_fill_node]) > 0:
                for ne in add_edges[min_fill_node]:
                    interaction.add_edge(ne[0], ne[1])

            # delete variable min_degree_node
            interaction.remove_node(min_fill_node)
            pi.append(min_fill_node)

        # print("Min FILL order PI = ", pi)
        return pi


# Mainly for trying things
def main():

    net_path = "testing/d_separation_example.BIFXML"

    reasoner = BNReasoner(net=net_path)

    reasoner.bn.draw_structure()


def main_martin():
    # net_path = "testing/abc_example.BIFXML"
    net_path = "testing/map_mpe_example.BIFXML"
    reasoner = BNReasoner(net=net_path)
    pi = ["J","I","X","Y","O"]
    reasoner.MPE({"J":True,"O":False},pi)
    # reasoner.marginal_distribution(["C"], {"A": True}, pi)

def main_debuging():
    net_path = "testing/psyc_disorders.BIFXML"
    reasoner = BNReasoner(net=net_path)

    pi = reasoner.ordering()
    print(pi)
    reasoner.marginal_distribution(["Autism", "OCD"], {"ADHD": False}, pi)
if __name__ == "__main__":
    main_debuging()
