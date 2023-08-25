# encoding: utf-8
"""
Partially based on work by:
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501, Market1501x2, Market1501x4
from .df1 import DF1

__factory = {
    "market1501": Market1501,
    "market1501x2": Market1501x2,
    "market1501x4": Market1501x4,
    "dukemtmcreid": DukeMTMCreID,
    "df1": DF1,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
