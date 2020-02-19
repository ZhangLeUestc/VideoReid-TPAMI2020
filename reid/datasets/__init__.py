from __future__ import absolute_import
from .market1501 import Market1501
from .cuhk03 import CUHK03
from .dukemtmc import DukeMTMC
from .ilidsvid import iLIDSVID
from .prid2011 import PRID2011
from .mars_train import Mars_train


__factory = {
    'cuhk03': CUHK03,
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'ilidsvid':iLIDSVID,
    'prid2011':PRID2011,
     'mars':Mars_train
}

def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)
