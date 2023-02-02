# -*- coding: utf-8 -*-
import sys
sys.path.extend(['.', '..'])
import time
import arrow
import networkx as nx

from src.generators import parse_seq
from src.kmernode2vec import KMerNode2Vec


def simulated_training_test():
    start_time = time.time()
    clf = KMerNode2Vec(p=1.0, q=0.001)
    clf.fit(
        graph=nx.DiGraph(),
        seqs=parse_seq(['../data_dir/input/demo/']),
        mer=8,
        path_to_edg_list_file='../data_dir/output/networkfile-demo.edg',
        path_to_embeddings_file='../data_dir/output/KMerNode2Vec-demo.txt',
    )
    print('total_time: ', time.time() - start_time)


def real_training_test():
    start_time = time.time()
    clf = KMerNode2Vec(p=1.0, q=0.001, workers=11)
    clf.fit(
        grxaph=nx.DiGraph(),
        seqs=parse_seq(['../data_dir/input/speed_dataset/1g/']),
        mer=8,
        path_to_edg_list_file=f"../data_dir/output/networkfile-{'1g'}-{arrow.utcnow().format('YYYYMMDD-HHmm')}.edg",
        path_to_embeddings_file=f"../data_dir/output/kmer-node2vec-{'1g'}-{arrow.utcnow().format('YYYYMMDD-HHmm')}.txt",
    )
    print('total_time: ', time.time() - start_time)


def main():
    # real_training_test()
    simulated_training_test()


if __name__ == '__main__':
    main()
