from math import factorial
from typing import Union, List, Tuple, Dict
import random
from numpy import empty, multiply
from BayesNet import BayesNet
from itertools import combinations, product
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
        if heuristic.lower() not in ["degree", "fill", "rand"]:
            raise Exception
        if heuristic == "degree":
            return self.__min_degree_order()
        elif heuristic == "fill":
            return self.__min_fill_order()
        elif heuristic == "rand":
            return self.__rand_order()

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

    def __get_compatible_cpts(self, E: Dict[str, bool]):
        S = []
        cpts = {}

        # self.__node_prune(Q, E)
        # self.__edge_prune(E)

        for var in self.bn.get_all_variables():
            cpts[var] = self.bn.get_cpt(var)
            if len(E) != 0:
                S.append(
                    self.bn.get_compatible_instantiations_table(pd.Series(E), cpts[var])
                )
            else:
                S.append(cpts[var])

        return S

    def marginal_distribution(self, Q: List[str], E: Dict[str, bool], pi: List[str]):
        self.network_pruning(Q, E)
        S = self.__get_compatible_cpts(E)

        query_joint_prob = self.joint_distribution(Q, S, pi)

        posterior_marginal_distribution = self.normalize(query_joint_prob)

        return posterior_marginal_distribution

    def MAP(self, Q: List[str], E: Dict[str, bool], pi: List[str]):

        # put the query variables in the end of pi
        pi_aux = deepcopy(pi)
        for o in pi_aux:
            if o in Q:
                pi.remove(o)
                pi.append(o)

        S = self.__get_compatible_cpts(E)

        for i in range(0, len(pi)):

            pi_i = pi[i]

            # get factors mentioning pi(i)
            factors_including_var = self.__get_factors_including_var(S, pi_i)

            if factors_including_var:
                # multiply all factors mentioning variable pi(i)
                f = self.multiply_factors(factors_including_var, pi_i)

                if pi_i in Q:
                    f_i = self.max_out_var(f, pi_i)
                else:
                    f_i = self.sum_out_var(f, pi_i)

                # remove elements factors_including_var from S
                for factor in factors_including_var:
                    arr = [
                        factor.sort_index()
                        .sort_index(axis=1)
                        .equals(s_factor.sort_index().sort_index(axis=1))
                        for s_factor in S
                    ]
                    for j in range(0, len(arr)):
                        if arr[j] == True:
                            S.pop(j)

                # then add new factor f_i to S
                S.append(f_i)

        S = self.multiply_factors(S, "")
        S = self.max_out_row(S)

        return S

    def MPE(self, E: Dict[str, bool], pi: List[str]):
        Q = self.bn.get_all_variables()

        S = self.__get_compatible_cpts(E)

        for i in range(0, len(pi)):

            pi_i = pi[i]

            # get factors mentioning pi(i)
            factors_including_var = self.__get_factors_including_var(S, pi_i)

            if factors_including_var:
                # multiply all factors mentioning variable pi(i)
                f = self.multiply_factors(factors_including_var, pi_i)

                f_i = self.max_out_var(f, pi_i)

                # remove elements factors_including_var from S
                for factor in factors_including_var:
                    arr = [
                        factor.sort_index()
                        .sort_index(axis=1)
                        .equals(s_factor.sort_index().sort_index(axis=1))
                        for s_factor in S
                    ]
                    for j in range(0, len(arr)):
                        if arr[j] == True:
                            S.pop(j)

                # then add new factor f_i to S
                S.append(f_i)

        S = self.multiply_factors(S, "")
        S = self.max_out_row(S)
        return S

    def joint_distribution(self, Q: List[str], S: List[pd.DataFrame], pi: List[str]):

        for i in range(0, len(pi)):

            pi_i = pi[i]

            # get factors mentioning pi(i)
            factors_including_var = self.__get_factors_including_var(S, pi_i)

            # multiply all factors mentioning variable pi(i)
            if factors_including_var:
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

        S = self.multiply_factors(S, "")

        return S

    def multiply_factors(self, factors: List[pd.DataFrame], var: str) -> pd.DataFrame:

        if len(factors) == 1:
            return factors[0]
        else:
            while len(factors) > 1:
                f1 = factors[0]
                f2 = factors[1]

                if (
                    not var
                ):  # this is for the last multiplication in case there is independent variables, so have tu multiply all independent factors
                    var = list(f1.columns)
                    var.remove("p")
                    mult = f1.merge(f2, how="cross")

                else:
                    var = list(
                        set(f1.columns) & set(f2.columns)
                    )  # get the common columns to do the merge
                    var.remove("p")

                    mult = f1.merge(f2, on=var)

                mult["p"] = mult.p_x * mult.p_y
                mult = mult.drop(["p_x", "p_y"], axis=1)

                factors = factors[2:]
                factors.append(mult)

        return factors[0]

    def sum_out_var(self, factor: pd.DataFrame, var: str) -> pd.DataFrame:

        variables = list(factor.columns)
        variables.remove("p")
        variables.remove(var)

        if len(variables) > 0:
            factor = factor.groupby(variables, as_index=False).agg("sum")

            factor = factor.drop([var], axis=1)

        return factor

    def max_out_var(self, factor: pd.DataFrame, var: str) -> pd.DataFrame:
        variables = list(factor.columns)
        variables.remove("p")
        variables.remove(var)

        if len(variables) > 0:

            df = factor
            df.reset_index(drop=True, inplace=True)
            df1 = df.groupby(variables, as_index=False)["p"].agg(["max", "idxmax"])

            df1["idxmax"] = df.loc[df1["idxmax"], var].values

            df1 = df1.rename(columns={"idxmax": var, "max": "p"}).reset_index()

            factor = df1

            return factor
        else:
            factor = factor.loc[factor["p"] == factor["p"].max()]

        return factor

    def max_out_row(self, factor: pd.DataFrame) -> pd.DataFrame:

        factor = factor.loc[factor["p"] == factor["p"].max()]

        return factor

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

    def __rand_order(self):
        variables = deepcopy(self.bn.get_all_variables())
        random.shuffle(variables)
        return variables

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


def use_case_questions():
    net_path = "testing/psyc_disorders.BIFXML"

    # Pr(Sex=F, Genetic=F, Depression=T, Anxiety=T, ADHD=T,Autism=F, OCD=F, SubstanceAbuse=F, EatingDisorder=F,Insomnia=F) = ?
    reasoner1 = BNReasoner(net=net_path)

    pi = reasoner1.ordering()
    q1 = reasoner1.marginal_distribution(
        [
            "Sex",
            "Genetic",
            "Depression",
            "Anxiety",
            "ADHD",
            "Autism",
            "OCD",
            "SA",
            "ED",
            "Insomnia",
        ],
        {},
        pi,
    )

    q1 = q1.loc[
        (q1["Sex"] == False)
        & (q1["Genetic"] == False)
        & (q1["Genetic"] == False)
        & (q1["Depression"] == True)
        & (q1["Anxiety"] == True)
        & (q1["ADHD"] == True)
        & (q1["Autism"] == False)
        & (q1["OCD"] == False)
        & (q1["SA"] == False)
        & (q1["ED"] == False)
        & (q1["Insomnia"] == False)
    ]
    print(
        "________________________________________________________________________________________________________________________________________________________________________"
    )
    print(
        "Q1: Pr(Sex=F, Genetic=F, Depression=T, Anxiety=T, ADHD=T,Autism=F, OCD=F, SubstanceAbuse=F, EatingDisorder=F,Insomnia=F) = {}\n".format(
            q1.iloc[0]["p"]
        )
    )
    print(q1)
    print(
        "________________________________________________________________________________________________________________________________________________________________________"
    )

    #  Pr(EatingDisorder=T, Insomnia=F |Depression=T) = ??
    reasoner2 = BNReasoner(net=net_path)
    q2 = reasoner1.marginal_distribution(["ED", "Insomnia"], {"Depression": True}, pi)
    print(q2)
    q2 = q2.loc[(q2["ED"] == True) & (q2["Insomnia"] == False)]

    print(
        "________________________________________________________________________________________________________________________________________________________________________"
    )
    print(
        "Q2: Pr(EatingDisorder=T, Insomnia=F |Depression=T) = {}\n".format(
            q2.iloc[0]["p"]
        )
    )
    print(q2)
    print(
        "________________________________________________________________________________________________________________________________________________________________________"
    )

    # Pr(Sex=T, OCD=T |ADHD=F, Autism=F) = ??
    reasoner3 = BNReasoner(net=net_path)
    q3 = reasoner3.marginal_distribution(
        ["Sex", "OCD"], {"ADHD": False, "Autism": False}, pi
    )
    print(q3)
    q3 = q3.loc[(q3["Sex"] == True) & (q3["OCD"] == True)]

    print(
        "________________________________________________________________________________________________________________________________________________________________________"
    )
    print("Q3: #Pr(Sex=T, OCD=T |ADHD=F, Autism=F) = {}\n".format(q3.iloc[0]["p"]))
    print(q3)
    print(
        "________________________________________________________________________________________________________________________________________________________________________"
    )

    # MAP Q= {SubstanceAbuse}, given  Genetics=T, Anxiety=T, Insomnia=F
    reasoner4 = BNReasoner(net=net_path)
    q4 = reasoner4.MAP(["SA"], {"Genetics": True, "Anxiety": True}, pi)
    print(q4)
    print(
        "________________________________________________________________________________________________________________________________________________________________________"
    )

    # What is the MAP of someone having substance abuse, given that person has the family history, anxiety, but no insomnia?
    reasoner5 = BNReasoner(net=net_path)
    q5 = reasoner5.MPE({"Genetics": True, "Anxiety": True, "Insomnia": False}, pi)
    print(q5)
    print(
        "________________________________________________________________________________________________________________________________________________________________________"
    )


if __name__ == "__main__":
    use_case_questions()
