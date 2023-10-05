# encoding: utf-8
"""
Partially based on work by:
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .prw import PRW
from .mot17 import MOT17


__factory = {
    "market1501": Market1501,
    "dukemtmcreid": DukeMTMCreID,
    "PRW": PRW,
    "MOT17": MOT17
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
