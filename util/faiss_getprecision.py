# -*- coding: utf-8 -*-
import sys
sys.path.extend(['.', '..'])
import os
import time
import faiss
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from pecanpy.wrappers import Timer


def ptr_to_file(
    kmer2vec_method='kmg2vec',
    seg2vec_method='AVG',
    input_file_dir='../data_dir/output/seg2vec/viral/',
):
    """ Return segments & subsegments with their vectors.

    Note:
        segments and subsegments are actually the same.
    """

    if not os.path.exists(input_file_dir):
        raise ValueError(input_file_dir + ' does not exist!')

    try:
        vec_files = []
        seg_name = org_subseg_name = None
        for root, dirs, files in os.walk(input_file_dir):
            for file in files:
                if file.endswith('.txt'):
                    if 'Vectors' in file and kmer2vec_method in file and \
                            seg2vec_method in file:
                        vec_files.append(os.path.join(root, file))
                    if 'SegmentNames' in file and '150bp' in file:
                        seg_name = os.path.join(root, file)
                    if 'random' in file and 'OriginalSegmentNames' in file:
                        org_subseg_name = os.path.join(root, file)
        try:
            assert len(vec_files) == 2
            for file in vec_files:
                if '-SegmentVectors' in file:
                    seg_vec = file
                if '-SubSegmentVectors' in file:
                    subseg_vec = file
            assert seg_vec is not None and subseg_vec is not None
        except ValueError:
            print('files fail to match')

    except IOError:
        print('files not found or fail to read')

    return seg_name, seg_vec, org_subseg_name, subseg_vec


@Timer('create faiss index')
def create_index(
    path_to_seg_vec=None,
    path_to_faiss_idx=None,
    dimension=128,
    method='HNSW',
    nlist=1000,
    nprobe=150,
    vertex_connection=100,
    ef_search=2000,
    ef_construction=128,
):
    """ Create index file for faiss.

    Args:
        path_to_seg_vec (str) : vectors to feed faiss.
            (default = None).
        path_to_faiss_idx str) : file that store vector indexes for faiss
            (default = "../data_dir/output/topKN/{'faiss-idx'}-{'dna2vec'}-{'AVG'}").
        dimension (int) : the input vector size
            (default = 128).
        method (str) : method to create faiss index, 'HNSW' or 'IVF' or
            'BRUTAL' or 'IVF_HNSW'
            (default = 'HNSW').
        nlist (int) : number of cells/clusters to partition data into
            (default = 1000).
        nprobe (int) : set how many of nearest cells to search
            (default = 150).
        vertex_connection (int) : number of connections each vertex will have
            (default = 150).
        ef_search (int) : depth of layers explored during search
            (default = 2000).
        ef_construction (int) : depth of layers explored during index construction
            (default = 128)

    """
    if os.path.isfile(path_to_faiss_idx):  # prevent overriding
        return True

    # read vectors
    xb = np.loadtxt(path_to_seg_vec)
    xb = xb.astype(np.float32)

    if method == 'IVF':
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dimension),
            dimension,
            nlist,
            faiss.METRIC_L2,
        )
        index.train(xb)
        index.add(xb)
        index.nprobe = nprobe

    elif method == 'HNSW':
        """
        efConstruction: The larger the setting, the higher the quality 
            of the construction graph and the higher the precision of the search, 
            but at the same time the indexing time becomes longer, 
            the recommended range is 100-2000
        efSearch: The larger the setting, the higher the recall rate, 
            but the longer the query response time. The recommended range 
            is 100-2000. In HNSW, the parameter ef is the abbreviation of efSearch
        M: Within a certain access, the larger the setting, the higher 
            the recall rate and the shorter the query response time, but at the same time, 
            the increase of M will lead to an increase in the indexing time. 
            The recommended range is 5-100.
        """
        index = faiss.IndexHNSWFlat(dimension, vertex_connection)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        index.add(xb)

    elif method == 'BRUTAL':
        index = faiss.IndexFlatL2(dimension)
        index.add(xb)

    elif method == 'IVF_HNSW':
        index = faiss.index_factory(
            dimension,
            f"IVF{nlist}_HNSW{vertex_connection},Flat"
        )

        index.nprobe = nprobe
        faiss.downcast_index(index.quantizer).hnsw.efSearch = ef_search
        faiss.downcast_index(index.quantizer).hnsw.efConstruction = ef_construction

        index.train(xb)
        index.add(xb)

    else:
        raise ValueError(
            "Only support for 'IVF', 'HNSW', 'BRUTAL' and 'IVF_HNSW'.\n"
        )

    faiss.write_index(index, path_to_faiss_idx)
    print(
        f"\n\n*** TopKN-Begin ***\n"
        f"The total of index: {str(index.ntotal)}\n"
        f"The index file(s) at：{path_to_faiss_idx}\n"
    )

    return index.is_trained


@Timer('make search for each query')
def getI(idx, xq, Topk):
    """ Get the index list of the most similar TopK
        for each retrieval query """
    D, I = idx.search(xq, Topk)
    return I


def compare(obj1, obj2):
    """ Compare whether the objectives corresponding
        to two vectors are consistent  """
    return 1 if obj1 == obj2 else 0


def print_precision(molecular_nums, denominator):
    """ output the precision table to the screen """
    for i in range(len(molecular_nums)):
        molecular_nums[i] /= denominator

    table = PrettyTable(['ID', 'Precision'])
    for i in range(len(molecular_nums)):
        table.add_row(
            [f"TopK{str(i+1)}", molecular_nums[i]]
        )
    print(table)


@Timer('run precision\n\n')
def precision(
    path_to_subseg_vec=None,
    path_to_subseg_name='../data_dir/output/seg2vec/SegmentNames-150bp.txt',
    path_to_seg_name='../data_dir/output/seg2vec/SegmentNames-150bp.txt',
    path_to_faiss_idx=f"../data_dir/output/topKN/{'faiss-idx'}-{'dna2vec'}-{'AVG'}",
    top_kn=20,
):
    """
    Compare the similarity between segments and sub-segments

    Note:
        path_to_seg_name is actually the same as path_to_subseg_name.
    """

    # define top-kn molecular
    molecular_lst = [0] * top_kn

    # sub-segment vectors for query
    xq = np.loadtxt(path_to_subseg_vec)
    xq = xq.astype(np.float32)

    # compare two groups of DNA segments
    obj1 = np.loadtxt(path_to_subseg_name, dtype=np.str_)
    obj2 = np.loadtxt(path_to_seg_name, dtype=np.str_)
    print(f"faiss stores {len(obj2)} vectors")

    # read index
    start_time = time.time()
    new_index = faiss.read_index(path_to_faiss_idx)
    new_index.hnsw.efSearch = 2000  # when faiss idx == HNSW
    print(f"loading faiss index cost {time.time() - start_time:.6f}s")

    # index list most similar to TopK
    I = getI(new_index, xq, Topk=top_kn)
    print(f"make successful search for {len(I)} queries")

    # TopKN
    for i in tqdm(range(len(I)), desc='precision'):
        for j in range(len(I[i])):
            if compare(obj1[i], obj2[I[i][j]]) == 1:
                """
                molecular_x += 1 when x >= j + 1 as topK is j + 1
                molecular_lst[x-1] = molecular_x
                """
                for k in range(j, len(molecular_lst)):
                    molecular_lst[k] += 1
                break

    time.sleep(1)
    print_precision(molecular_lst, len(I))


def retrieval_ananlysis(
    path_to_subseg_vec: str,
    path_to_subseg_name: str,
    path_to_seg_name: str,
    path_to_faiss_indexes: list,  # 一个列表包含多个不同k下对应的k-mer embedding对应的索引文件
    max_top_kn: int = 20,
    top_kn_border: list = [5, 10, 15, 20]
):
    print(f"**** retrieval_ananlysis begins ****")

    # sub-segment vectors for query
    xq = np.loadtxt(path_to_subseg_vec)
    xq = xq.astype(np.float32)

    # compare two groups of DNA segments
    obj1 = np.loadtxt(path_to_subseg_name, dtype=np.str_)
    obj2 = np.loadtxt(path_to_seg_name, dtype=np.str_)
    print(f"faiss stores {len(obj2)} vectors")

    for path_to_faiss_idx in path_to_faiss_indexes:
        # read index
        start_time = time.time()
        new_index = faiss.read_index(path_to_faiss_idx)
        new_index.hnsw.efSearch = 2000  # when faiss idx == HNSW
        print(f"loading faiss index cost {time.time() - start_time:.6f}s")

        # given a certain TopK level, we study which queries to be retrieved successfully or not
        """
        1) 找出难检索序列：检查哪些序列在topk水平下总是检索不出来
        2) (重点)找出同一个query在相同(最大)topk、但不同k-mer的水平下的answers差异
        """
        print("**** find retrieved-hard sequences ****")
        for tk in top_kn_border:
            # index list most similar to TopK
            I = getI(new_index, xq, Topk=tk)

            retrieved_queries = set()
            unretrieved_queries = set()
            for i in tqdm(range(len(I)), desc='precision'):
                retrieved_flag = 0  # to signal if obj[i] were successfully retrieved
                for j in range(len(I[i])):
                    if compare(obj1[i], obj2[I[i][j]]) == 1:
                        retrieved_flag = 1
                if retrieved_flag == 0:
                    unretrieved_queries.add(obj1[i])
                else:
                    retrieved_queries.add(obj1[i])

            for query in retrieved_queries:
                # tk illustrates the topk level
                print("faiss index name:"+path_to_faiss_idx, "topk level:"+tk, "state:retrieved", "query:"+query)
            for query in unretrieved_queries:
                print("faiss index name:"+path_to_faiss_idx, "topk level:"+tk, "state:unretrieved", "query:"+query)

        print("**** find difference between given query's answers under different k-mer embedding ****")













