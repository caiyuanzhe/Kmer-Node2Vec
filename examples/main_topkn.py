# -*- coding: utf-8 -*-
import sys
sys.path.extend(['.', '..'])
import arrow
from icecream import ic

from util.faiss_getprecision import ptr_to_file
from util.faiss_getprecision import create_index
from util.faiss_getprecision import precision
from util.log import Tee


def single_test():
    """ 设置关键文件路径 """
    species = 'mixed'  # 'viral' or 'bacteria' or 'fungi' or 'mixed'
    kmer2vec_method = 'dna2vec'  # 'kmg2vec' or 'dna2vec'
    seg2vec_method = 'AVG'  # 'AVG' or 'TFIDF' or 'SIF'
    index_method = 'HNSW'
    faiss_idx_path = f"../data_dir/output/topKN/faiss-idx-{species}-{kmer2vec_method}-{seg2vec_method}-{index_method}"

    # set a log for recording the result
    logger = Tee(
        f"../data_dir/output/topKN/ret-"
        f"{species}-{kmer2vec_method}-{seg2vec_method}-{index_method}-{arrow.utcnow().format('YYYYMMDD-HHmm')}.log"
    )
    sys.stdout = logger

    seg_name_path, seg_vec_path, subseg_name_path, subseg_vec_path = ptr_to_file(
        kmer2vec_method=kmer2vec_method,
        seg2vec_method=seg2vec_method,
        input_file_dir=f"../data_dir/output/seg2vec/{species}/",
    )
    ic(seg_name_path, seg_vec_path, subseg_name_path, subseg_vec_path)

    if create_index(
        seg_vec_path,
        faiss_idx_path,
        dimension=128,
        method=index_method,
        vertex_connection=100,
        ef_search=2000,
        ef_construction=128,
    ):
        precision(subseg_vec_path, subseg_name_path, seg_name_path, faiss_idx_path, top_kn=20)


def multi_test():
    """ 设置关键文件路径 """
    species = 'mixed'  # 'viral' or 'bacteria' or 'fungi' or 'mixed'
    kmer2vec_method = ['kmg2vec', 'dna2vec']
    seg2vec_method = ['SIF', 'AVG', 'TFIDF']
    index_method = 'HNSW'

    # set a log for recording the result
    logger = Tee(f"../data_dir/output/topKN/result-{species}-{arrow.utcnow().format('YYYYMMDD-HHmm')}.log")
    sys.stdout = logger

    for i in range(len(kmer2vec_method)):
        for j in range(len(seg2vec_method)):

            faiss_idx_path = f"../data_dir/output/topKN/faiss-idx-" \
                             f"{species}-{kmer2vec_method[i]}-{seg2vec_method[j]}-{index_method}"

            seg_name_path, seg_vec_path, subseg_name_path, subseg_vec_path = ptr_to_file(
                kmer2vec_method=kmer2vec_method[i],
                seg2vec_method=seg2vec_method[j],
                input_file_dir=f"../data_dir/output/seg2vec/{species}/",
            )
            ic(seg_name_path, seg_vec_path, subseg_name_path, subseg_vec_path)

            if create_index(
                seg_vec_path,
                faiss_idx_path,
                dimension=128,
                method=index_method,
                vertex_connection=100,
                ef_search=2000,
                ef_construction=128,
            ):
                precision(subseg_vec_path, subseg_name_path, seg_name_path, faiss_idx_path, top_kn=20)


if __name__ == '__main__':
    single_test()
