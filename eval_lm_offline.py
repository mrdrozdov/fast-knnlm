"""
# First, save knns.
python rerank_lib/eval_lm_offline.py --save-knns

# (optional) Save exact dist.
python rerank_lib/eval_lm_offline.py --save-exact --load-dstore-in-mem

# You can also do test.
python rerank_lib/eval_lm_offline.py --save-knns --test

# Run eval for validation.
python rerank_lib/eval_lm_offline.py
# or
python rerank_lib/eval_lm_offline.py --exact
"""


import argparse
import collections
import json
import os
import sys

import faiss
import numpy as np
import torch

from tqdm import tqdm


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dstore', default='/iesl/local/adrozdov/knnlm_data', type=str)
    parser.add_argument('--dstore-size', default=103225485, type=int)
    parser.add_argument('--eval-dstore', default='/iesl/local/adrozdov/knnlm_data.valid', type=str)
    parser.add_argument('--eval-dstore-size', default=217646, type=int)
    parser.add_argument('--eval-dstore-cache', default='/iesl/local/adrozdov/knnlm_data.valid.cache', type=str)
    parser.add_argument('--k', default=1024)

    parser.add_argument('--load-dstore-in-mem', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--save-knns', action='store_true')
    parser.add_argument('--save-exact', action='store_true')

    return parser


class Dstore(object):
    def __init__(self, args):
        path = args.dstore
        dstore_size = args.dstore_size

        #self.sim_func = None if args.exact else 'do_not_recomp_l2'
        self.sim_func = 'do_not_recomp_l2'
        self.k = 1024

        self.keys = np.memmap(f'{path}/dstore_keys.npy', dtype=np.float16, mode='r', shape=(dstore_size, 1024))
        self.vals = np.memmap(f'{path}/dstore_vals.npy', dtype=np.int16, mode='r', shape=(dstore_size, 1))

        if args.load_dstore_in_mem:
            batch_size = 1024 * 10
            for k in ['keys', 'vals']:
                v = getattr(self, k)
                print(f'alloc {k}')
                new_v = np.empty(v.shape, dtype=v.dtype)
                print(f'load {k}')
                for start in tqdm(range(0, v.shape[0], batch_size), desc=f'load {k}'):
                    end = min(start + batch_size, v.shape[0])
                    new_v[start:end] = v[start:end]
                setattr(self, k, new_v)

        print('load index')
        indexfile = f'{path}/knn.index'
        self.index = faiss.read_index(indexfile, faiss.IO_FLAG_ONDISK_SAME_DIR)

        self.half = True
        self.metric_type = 'l2'

    def combine_knn_and_vocab_probs(self, knn_p, vocab_p, coeff):
        combine_probs = torch.stack([vocab_p, knn_p], dim=0)
        coeffs = torch.ones_like(combine_probs)
        coeffs[0] = np.log(1 - coeff)
        coeffs[1] = np.log(coeff)
        curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)

        return curr_prob

    def get_knns(self, query):
        if query.dtype == np.float16:
            query = query.astype(np.float32)
        dists, knns = self.index.search(query, self.k)
        return dists, knns

    def get_knn_log_prob_from_knns(self, knns, dists, target):
        d = torch.from_numpy(dists).float()
        probs = torch.log_softmax(d, -1)

        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().squeeze(-1), torch.from_numpy(target).long()).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        log_prob = torch.logsumexp(probs + index_mask, dim=-1, keepdim=True)

        return log_prob

    def get_knn_log_prob(self, query, target):
        def dist_func(d, k, q, function=None):
            if not function:
                # Default behavior for L2 metric is to recompute distances.
                # Default behavior for IP metric is to return faiss distances.
                qsize = q.shape
                if self.metric_type == 'l2':
                    knns_vecs = torch.from_numpy(self.keys[k]).cuda().view(qsize[0], self.k, -1)
                    q = torch.from_numpy(q).cuda()
                    if self.half:
                        knns_vecs = knns_vecs.half()
                        q = q.half()
                    query_vecs = q.view(qsize[0], 1, qsize[1]).repeat(1, self.k, 1)
                    l2 = torch.sum((query_vecs - knns_vecs.detach())**2, dim=2)
                    return -1 * l2
                return d

            if function == 'dot':
                qsize = q.shape
                return (torch.from_numpy(self.keys[k]) * q.view(qsize[0], 1, qsize[1])).sum(dim=-1)

            if function == 'do_not_recomp_l2':
                return -1 * d

            raise ValueError("Invalid knn similarity function!")

        # queries  are TxBxC
        # reshape: (TxB)xC
        dists, knns = self.get_knns(query)
        # (T_reducedxB)xK
        dists = torch.from_numpy(dists).cuda()
        dists = dist_func(dists, knns, query, function=self.sim_func)
        probs = torch.log_softmax(dists, dim=-1)

        index_mask = torch.eq(torch.from_numpy(self.vals[knns]).long().cuda().squeeze(-1), torch.from_numpy(target).long().cuda()).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        # (T_reducedxB)
        yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1)

        output = {}
        output['log_prob'] = yhat_knn_prob
        output['knns'] = knns
        output['dists'] = dists

        return output


class Dataset(object):
    def __init__(self, args):
        self.args = args
        path = args.eval_dstore
        dstore_size = args.eval_dstore_size
        self.query = np.memmap(f'{path}/dstore_keys.npy', dtype=np.float16, mode='r', shape=(dstore_size, 1024))
        self.target = np.memmap(f'{path}/dstore_vals.npy', dtype=np.int16, mode='r', shape=(dstore_size, 1))
        self.prob = np.memmap(f'{path}/dstore_prob.npy', dtype=np.float16, mode='r', shape=(dstore_size, 1))

        for k in ['query', 'target', 'prob']:
            v = getattr(self, k)
            new_v = np.ones(v.shape, dtype=v.dtype)
            new_v[:] = v
            setattr(self, k, new_v)

    def load_cache(self):
        args = self.args
        path = args.eval_dstore_cache
        dstore_size = args.eval_dstore_size
        self.dists = np.memmap(f'{path}/dstore_cache_dists.npy', dtype=np.float32, mode='r', shape=(dstore_size, 1024))
        self.knns = np.memmap(f'{path}/dstore_cache_knns.npy', dtype=np.int, mode='r', shape=(dstore_size, 1024))

    def load_exact_dist(self):
        args = self.args
        path = args.eval_dstore_cache
        dstore_size = args.eval_dstore_size
        filename = f'{path}/dstore_cache_exact_dists.npy'
        assert os.path.exists(filename)
        self.exact_dists = np.memmap(filename, dtype=np.float32, mode='r', shape=(dstore_size, 1024))


def eval_ppl(p):
    return 2**(-p.mean()/np.log(2))


def main(args):
    print('load dataset')
    dataset = Dataset(args)
    print('load dstore')
    dstore = Dstore(args)

    if args.save_knns:
        cache = collections.defaultdict(list)

        batch_size = 128

        for start in tqdm(range(0, dataset.query.shape[0], batch_size)):
            end = min(start + batch_size, dataset.query.shape[0])

            query, target = dataset.query[start:end], dataset.target[start:end]
            lm_prob = torch.from_numpy(dataset.prob[start:end]).float().view(-1, 1).cuda()
            dists, knns = dstore.get_knns(query)
            cache['dists'].append(dists)
            cache['knns'].append(knns)

        os.system(f'mkdir -p {args.eval_dstore_cache}')

        dists = np.concatenate(cache['dists'], 0)
        knns = np.concatenate(cache['knns'], 0)

        dstore_dists = np.memmap(f'{args.eval_dstore_cache}/dstore_cache_dists.npy', dtype=np.float32, mode='w+', shape=dists.shape)
        dstore_dists[:] = dists
        dstore_knns = np.memmap(f'{args.eval_dstore_cache}/dstore_cache_knns.npy', dtype=np.int, mode='w+', shape=knns.shape)
        dstore_knns[:] = knns

        print('done')
        sys.exit()

    dataset.load_cache()

    keys = dstore.keys
    vals = dstore.vals
    query = dataset.query
    target = dataset.target
    knns = dataset.knns
    scores = dataset.dists

    if args.save_exact:
        new_dist = np.ones(scores.shape, dtype=scores.dtype)

        batch_size = 128

        # TODO: GPU usage is low. Try using dataloader?
        for start in tqdm(range(0, query.shape[0], batch_size), desc='exact'):
            end = min(start + batch_size, query.shape[0])

            q_vecs = torch.from_numpy(query[start:end]).float().cuda()
            k_idx = knns[start:end]
            k_vecs = torch.from_numpy(keys[k_idx]).float().cuda()
            d = -torch.sum((q_vecs[:, None, :] - k_vecs)**2, 2)

            new_dist[start:end] = d.cpu().numpy()

        dstore_exact_dists = np.memmap(f'{args.eval_dstore_cache}/dstore_cache_exact_dists.npy', dtype=np.float32, mode='w+', shape=score.shape)
        dstore_exact_dists[:] = new_dist

        print('done')
        sys.exit()

    if args.exact:
        dstore.load_exact_dists()
        dists = dstore.exact_dists
    else:
        dists = -1 * scores

    def get_knn_prob():
        d = torch.from_numpy(dists).float()
        probs = torch.log_softmax(d, -1)

        index_mask = torch.eq(torch.from_numpy(dstore.vals[knns]).long().squeeze(-1), torch.from_numpy(target).long()).float()
        index_mask[index_mask == 0] = -10000 # for stability
        index_mask[index_mask == 1] = 0

        log_prob = torch.logsumexp(probs + index_mask, dim=-1)

        return log_prob

    print(knns.shape)

    knn_prob = get_knn_prob().view(-1, 1)
    lm_prob = torch.from_numpy(dataset.prob).float()
    new_prob = dstore.combine_knn_and_vocab_probs(knn_prob, lm_prob, 0.25)

    ppl = eval_ppl(new_prob)
    print(f'ppl = {ppl}')


if __name__ == '__main__':
    args = argument_parser().parse_args()

    if args.test:
        args.eval_dstore = '/iesl/local/adrozdov/knnlm_data.test'
        args.eval_dstore_cache = '/iesl/local/adrozdov/knnlm_data.test.cache'
        args.eval_dstore_size = 245569

    print(json.dumps(args.__dict__))

    with torch.no_grad():
        main(args)

