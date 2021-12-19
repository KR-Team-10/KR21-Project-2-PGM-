import pytest
from BNReasoner import BNReasoner


def test_d_separation():
    net = "testing/d_separation_example.BIFXML"
    reasoner = BNReasoner(net)
    reasoner.bn.draw_structure()

    assert reasoner.d_separation(X=["T", "A"], Z=["S", "B"], Y=["P", "D"]) == True
