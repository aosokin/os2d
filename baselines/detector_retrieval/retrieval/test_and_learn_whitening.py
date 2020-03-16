# Started the training script from 
# https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/v1.1/cirtorch/examples/test.py

import argparse
import os
import time
import pickle
import pdb

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
# from cirtorch.datasets.testdataset import configdataset
from retrieval_data import configdataset
from cirtorch.utils.download import download_train, download_test
from cirtorch.utils.whiten import whitenlearn, whitenapply
# from cirtorch.utils.evaluate import compute_map_and_print
from retrieval_data import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help="network path, destination where network is saved")
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help="network off-the-shelf, in the format 'ARCHITECTURE-POOLING' or 'ARCHITECTURE-POOLING-{reg-lwhiten-whiten}'," + 
                        " examples: 'resnet101-gem' | 'resnet101-gem-reg' | 'resnet101-gem-whiten' | 'resnet101-gem-lwhiten' | 'resnet101-gem-reg-whiten'")

# test options
parser.add_argument('--datasets', '-d',
                    help="comma separated list of test datasets")
parser.add_argument('--image-size', '-imsize', default=1024, type=int,
                    help="maximum size of longer image side used for testing (default: 1024)")
parser.add_argument('--multiscale', '-ms', default='[1, 2**(1/2), 1/2**(1/2)]', 
                    help="use multiscale vectors for testing, " + 
                    " examples: '[1]' | '[1, 1/2**(1/2), 1/2]' | '[1, 2**(1/2), 1/2**(1/2)]' (default: '[1, 2**(1/2), 1/2**(1/2)]')")
parser.add_argument('--whitening', '-w', default=None,
                    help="dataset used to learn whitening for testing (default: None)")

def main():
    args = parser.parse_args()

    # loading network from path
    if args.network_path is not None:

        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        if args.network_path in PRETRAINED:
            # pretrained networks (downloaded automatically)
            state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(get_data_root(), 'networks'))
        else:
            # fine-tuned network from path
            state = torch.load(args.network_path)

        # parsing net params from meta
        # architecture, pooling, mean, std required
        # the rest has default values, in case that is doesnt exist
        net_params = {}
        net_params['architecture'] = state['meta']['architecture']
        net_params['pooling'] = state['meta']['pooling']
        net_params['local_whitening'] = state['meta'].get('local_whitening', False)
        net_params['regional'] = state['meta'].get('regional', False)
        net_params['whitening'] = state['meta'].get('whitening', False)
        net_params['mean'] = state['meta']['mean']
        net_params['std'] = state['meta']['std']
        net_params['pretrained'] = False

        # load network
        net = init_network(net_params)
        net.load_state_dict(state['state_dict'])
        
        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']
        
        print(">>>> loaded network: ")
        if "epoch" in state:
            print("Model after {} epochs".format(state["epoch"]))
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:

        # parse off-the-shelf parameters
        offtheshelf = args.network_offtheshelf.split('-')
        net_params = {}
        net_params['architecture'] = offtheshelf[0]
        net_params['pooling'] = offtheshelf[1]
        net_params['local_whitening'] = 'lwhiten' in offtheshelf[2:]
        net_params['regional'] = 'reg' in offtheshelf[2:]
        net_params['whitening'] = 'whiten' in offtheshelf[2:]
        net_params['pretrained'] = True

        # load off-the-shelf network
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(net_params)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters: test both single scale and multiscale
    ms_singlescale = [1]
    msp_singlescale = 1

    ms_multiscale = list(eval(args.multiscale))
    msp_multiscale = 1
    if len(ms_multiscale)>1 and net.meta['pooling'] == 'gem' and not net.meta['regional'] and not net.meta['whitening']:
        msp_multiscale = net.pool.p.item()
    print(">> Set-up multiscale:")
    print(">>>> ms: {}".format(ms_multiscale))
    print(">>>> msp: {}".format(msp_multiscale))

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.whitening is not None:
        start = time.time()
        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
            Lw = net.meta['Lw'][args.whitening]
        else:
           # if we evaluate networks from path we should save/load whitening
            # not to compute it every time
            if args.network_path is not None:
                whiten_fn = args.network_path + '_{}_whiten'.format(args.whitening)
                whiten_fn += '.pth'
            else:
                whiten_fn = None

            if whiten_fn is not None and os.path.isfile(whiten_fn):
                print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
                Lw = torch.load(whiten_fn)
            else:
                Lw = {}
                for whiten_type, ms, msp in zip(["ss", "ms"], [ms_singlescale, ms_multiscale], [msp_singlescale, msp_multiscale]):
                    print('>> {0}: Learning whitening {1}...'.format(args.whitening, whiten_type))

                    # loading db
                    db_root = os.path.join(get_data_root(), 'train', args.whitening)
                    ims_root = os.path.join(db_root, 'ims')
                    db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
                    with open(db_fn, 'rb') as f:
                        db = pickle.load(f)
                    images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

                    # extract whitening vectors
                    print('>> {}: Extracting...'.format(args.whitening))
                    wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)

                    # learning whitening 
                    print('>> {}: Learning...'.format(args.whitening))
                    wvecs = wvecs.numpy()
                    m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
                    Lw[whiten_type] = {'m': m, 'P': P}

                    print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time()-start)))

                # saving whitening if whiten_fn exists
                if whiten_fn is not None:
                    print('>> {}: Saving to {}...'.format(args.whitening, whiten_fn))
                    torch.save(Lw, whiten_fn)
    else:
        Lw = None

    # evaluate on test datasets
    datasets = args.datasets.split(',')
    for dataset in datasets: 
        start = time.time()

        for whiten_type, ms, msp in zip(["ss", "ms"], [ms_singlescale, ms_multiscale], [msp_singlescale, msp_multiscale]):
            print('>> Extracting feature on {0}, whitening {1}'.format(dataset, whiten_type))

            # prepare config structure for the test dataset
            cfg = configdataset(dataset, os.path.join(get_data_root(), 'test'))
            images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
            qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
            bbxs = [tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])]

            # extract database and query vectors
            print('>> {}: database images...'.format(dataset))
            vecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
            print('>> {}: query images...'.format(dataset))
            qvecs = extract_vectors(net, qimages, args.image_size, transform, bbxs=bbxs, ms=ms, msp=msp)
            
            print('>> {}: Evaluating...'.format(dataset))

            # convert to numpy
            vecs = vecs.numpy()
            qvecs = qvecs.numpy()

            # search, rank, and print
            scores = np.dot(vecs.T, qvecs)
            ranks = np.argsort(-scores, axis=0)
            compute_map_and_print(dataset, ranks, cfg['gnd'])

            if Lw is not None:
                # whiten the vectors
                vecs_lw  = whitenapply(vecs, Lw[whiten_type]['m'], Lw[whiten_type]['P'])
                qvecs_lw = whitenapply(qvecs, Lw[whiten_type]['m'], Lw[whiten_type]['P'])

                # search, rank, and print
                scores = np.dot(vecs_lw.T, qvecs_lw)
                ranks = np.argsort(-scores, axis=0)
                compute_map_and_print(dataset + ' + whiten {}'.format(whiten_type), ranks, cfg['gnd'])

            print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()
