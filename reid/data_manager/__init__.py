from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from .mars_train import Mars_train
from .mars_test import Mars_test
from .ilidsvid import iLIDSVID
from .prid2011 import PRID2011






__vidreid_factory_train = {
    'mars': Mars_train,
    'ilidsvid': iLIDSVID,
    'prid2011': PRID2011,
    
}
__vidreid_factory_test = {
    'mars': Mars_test,
    'ilidsvid': iLIDSVID,
    'prid2011': PRID2011,
    
}


def get_names():
    return list(__imgreid_factory.keys()) + list(__vidreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)


def init_vidreid_train_dataset(name, **kwargs):
    if name not in list(__vidreid_factory_train.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory_train.keys())))
    return __vidreid_factory_train[name](**kwargs)

def init_vidreid_test_dataset(name, **kwargs):
    if name not in list(__vidreid_factory_test.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory_test.keys())))
    return __vidreid_factory_test[name](**kwargs)
