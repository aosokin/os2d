# Code partially copied from https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/v1.1/cirtorch/datasets/traindataset.py

import os
import pickle
import numpy as np
from functools import lru_cache

from cirtorch.datasets.traindataset import TuplesDataset as TuplesDatasetOriginal
from cirtorch.utils.general import get_data_root
from cirtorch.datasets.datahelpers import default_loader, cid2filename
from cirtorch.datasets.testdataset import config_imname, config_qimname
from cirtorch.utils.evaluate import compute_map


@lru_cache()
def loader_hashed(path):
    return default_loader(path)


def configdataset(dataset, dir_main):
    """
    Fucntion started from https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/v1.1/cirtorch/datasets/testdataset.py
    """
    # loading imlist, qimlist, and gnd, in cfg as a dict
    gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
    with open(gnd_fname, 'rb') as f:
        cfg = pickle.load(f)
    cfg['gnd_fname'] = gnd_fname

    cfg['ext'] = '.jpg'
    cfg['qext'] = '.jpg'
    cfg['dir_data'] = os.path.join(dir_main, dataset)
    cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')

    cfg['n'] = len(cfg['imlist'])
    cfg['nq'] = len(cfg['qimlist'])

    cfg['im_fname'] = config_imname
    cfg['qim_fname'] = config_qimname

    cfg['dataset'] = dataset

    return cfg


class TuplesDataset(TuplesDatasetOriginal):
    """
    Inheriting from TuplesDataset from
    https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/v1.1/cirtorch/datasets/traindataset.py
    Need to add my own data
    """
    def __init__(self, name, mode, imsize=None, nnum=5, qsize=2000, poolsize=20000, transform=None, loader=loader_hashed):

        if not (mode == 'train' or mode == 'val'):
            raise RuntimeError("MODE should be either train or val, passed as string")

        # setting up paths
        data_root = get_data_root()
        db_root = os.path.join(data_root, 'train', name)
        ims_root = os.path.join(db_root, 'ims')

        # loading db
        db_fn = os.path.join(db_root, '{}.pkl'.format(name))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)[mode]

        # setting fullpath for images
        self.images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

        # initializing tuples dataset
        self.name = name
        self.mode = mode
        self.imsize = imsize
        self.clusters = db['cluster']
        self.qpool = db['qidxs']
        self.ppool = db['pidxs']

        ## If we want to keep only unique q-p pairs 
        ## However, ordering of pairs will change, although that is not important
        # qpidxs = list(set([(self.qidxs[i], self.pidxs[i]) for i in range(len(self.qidxs))]))
        # self.qidxs = [qpidxs[i][0] for i in range(len(qpidxs))]
        # self.pidxs = [qpidxs[i][1] for i in range(len(qpidxs))]

        # size of training subset for an epoch
        self.nnum = nnum
        self.qsize = min(qsize, len(self.qpool))
        self.poolsize = min(poolsize, len(self.images))
        self.qidxs = None
        self.pidxs = None
        self.nidxs = None

        self.transform = transform
        self.loader = loader

        self.print_freq = 10


def compute_map_and_print(dataset, ranks, gnd, kappas=[1, 5, 10]):
    """
    Function started from https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/v1.1/cirtorch/utils/evaluate.py
    """
    # new evaluation protocol
    if dataset.startswith('roxford5k') or dataset.startswith('rparis6k'):
        
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
            gnd_t.append(g)
        mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, kappas)

        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, kappas)

        print('>> {}: mAP E: {}, M: {}, H: {}'.format(dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
        print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(dataset, kappas, np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
        return mapM
    else:
        # old evaluation potocol
        map, aps, _, _ = compute_map(ranks, gnd)
        print('>> {}: mAP {:.2f}'.format(dataset, np.around(map*100, decimals=2)))
        return map
