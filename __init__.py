"""CoMeta reproduction package (ICDE paper: Collaborative Meta-Learning for
E-Taxi Charging Pricing in Heterogeneous Internet of Energy).

Reuses the DBCNet reproduction's city/data layer (roadnet/realdata) by
extending sys.path to the sibling ``dbcnet_repro`` directory; the datasets
themselves live in ``dbcnet_repro/data``.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DBCNET = os.path.normpath(os.path.join(_HERE, "..", "..", "dbcnet_repro"))
if _DBCNET not in sys.path:
    sys.path.insert(0, _DBCNET)
DATA_DIR = os.path.join(_DBCNET, "data")
