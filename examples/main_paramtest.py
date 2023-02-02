# -*- coding: utf-8 -*-
import sys
sys.path.extend(['.', '..'])
import time
import os
import arrow
import networkx as nx
from gensim.models import KeyedVectors
import arrow
from icecream import ic

from util.faiss_getprecision import create_index
from util.faiss_getprecision import precision
from util.perf_tools import Tee
from src.generators import seq2segs
from src.generators import seg2sentence
from src.generators import extract_seg
from src.generators import parse_seq
from src.kmernode2vec import KMerNode2Vec


class ParamsExploration:
    """
    This test is for exploration of setting params P and Q.
    """
    def __init__(self):
        """ Params and IO settings """
        """ k-mer embeddings """
        self.params = [(1.0, 0.001), (1.0, 0.25), (1.0, 1.0), (1.0, 4.0)]  # (p, q)

        self.kmer_emb_seqs = parse_seq(['../data_dir/input/precision_dataset/viral/'])
        self.kmer_vec_output_dir = f"../data_dir/output/seg2vec/{'params_test'}/"
        self.kmer_vec_files = list()

        """ segment embeddings """
        self.seg_emb_seqs = parse_seq(['../data_dir/input/precision_dataset/viral/'])
        self.seg_vec_output_dir = f"../data_dir/output/seg2vec/{'params_test'}/"  # dir to output segment embeddings
        self.seg_file = f"{self.seg_vec_output_dir}SegmentNames-150bp.txt"

        # randomly choose 1k segments from seg_file and extract subsegments from the 1k segments
        self.extracted_orgseg_file = f"{self.seg_vec_output_dir}random-1k-OriginalSegmentNames-150bp.txt"
        self.extracted_subseg_file = f"{self.seg_vec_output_dir}random-1k-SubSegmentNames-75bp.txt"

        """ segment embeddings """
        self.seg2vec_method = ['AVG', 'TFIDF', 'SIF',]  # ordered

        self.faiss_vec_input_dir = self.seg_vec_output_dir + 'SeqVec/'
        self.faiss_idx = f"{self.faiss_vec_input_dir}FaissIndex"
        self.faiss_log = f"{self.faiss_vec_input_dir}FaissLog-{arrow.utcnow().format('YYYYMMDD-HHmm')}.log"

    def kmer_embeddings(self):
        """ Obtain k-mer embeddings with different P and Q settings. """
        for ps in self.params:
            start_time = time.time()
            clf = KMerNode2Vec(p=ps[0], q=ps[1], workers=15)
            clf.fit(
                graph=nx.DiGraph(),
                seqs=self.kmer_emb_seqs,
                mer=8,
                path_to_edg_list_file=self.kmer_vec_output_dir + f"networkfile-p{ps[0]}-q{ps[1]}-{arrow.utcnow().format('YYYYMMDD-HHmm')}.edg",
                path_to_embeddings_file=self.kmer_vec_output_dir + f"kmer-node2vec-p{ps[0]}-q{ps[1]}-{arrow.utcnow().format('YYYYMMDD-HHmm')}.txt",
            )
            print(f"p{ps[0]}-q{ps[1]} total_time: ", time.time() - start_time)

    def segment_embeddings(self):
        """ Obtain segment embeddings with different pre-trained k-mer embeddings. """
        seq2segs(
            self.seg_emb_seqs,
            150,
            path_to_segs_file=self.seg_file,
        )

        with open(self.seg_file, 'r', encoding='utf-8') as fp:
            segs = [line.split('\n')[0] for line in fp.readlines()]
        sentences = seg2sentence(segs, 8)

        from util.vectorizer import SIF, AVG, TFIDF
        for w2v in self.kmer_vec_files:
            # k-mer vectors
            vecs = KeyedVectors.load_word2vec_format(w2v)
            print(f"k-mer vectors: {w2v}")

            clfs = [AVG(vecs), TFIDF(vecs), SIF(vecs, cache_size_gb=40)]
            for clf in clfs:
                clf.train(sentences)
                clf.save_embs_format(
                    self.seg_vec_output_dir,
                    f"{'SegmentVectors'}-{w2v.split('/')[-1].split('.txt')[0]}"
                )
                print('******')

    def generate_segs(self):
        if os.path.isfile(self.extracted_subseg_file):  # prevent overriding
            return self.extracted_subseg_file

        extract_seg(
            self.seg_file, 150, 1000,
            path_to_extracted_orgsegs_file=self.extracted_orgseg_file,
            path_to_extracted_subsegs_file=self.extracted_subseg_file,
        )
        print(f"segs_file: {self.seg_file}\n"
              f"orgsegs_file: {self.extracted_orgseg_file}\n"
              f"subsegs_file:{self.extracted_subseg_file}"
              )

    def subseg_embeddings(self):

        self.generate_segs()  # Randomly extract sub-segments from self.seg_file

        # load subsegments
        with open(self.extracted_subseg_file, 'r', encoding='utf-8') as fp:
            subsegs = [line.split('\n')[0] for line in fp.readlines()]
        sentences = seg2sentence(subsegs, 8)

        from util.vectorizer import SIF, AVG, TFIDF
        for w2v in self.kmer_vec_files:
            # k-mer2vec file
            vecs = KeyedVectors.load_word2vec_format(w2v)
            print(f"k-mer vectors: {w2v}")

            # calculate sentence vectors by averaging word vectors
            clfs = [AVG(vecs), TFIDF(vecs), SIF(vecs, cache_size_gb=42)]
            for clf in clfs:
                clf.train(sentences)
                clf.save_embs_format(
                    self.seg_vec_output_dir,
                    f"{'SubSegmentVectors'}-{w2v.split('/')[-1].split('.txt')[0]}"
                )

    def seg_and_subseg_embeddings(self):
        for root, dirs, files in os.walk(self.kmer_vec_output_dir):
            for file in files:
                if file.startswith('kmer-node2vec'):
                    self.kmer_vec_files.append(os.path.join(root, file))

        self.segment_embeddings()
        self.subseg_embeddings()

    @staticmethod
    def ptr_to_file(
        identify_flag,
        seg2vec_method='AVG',
        input_file_dir=f"../data_dir/output/seg2vec/{'params_test'}/{'SeqVec'}/",
    ):
        """ Return segments & subsegments with their vectors.

        Note:
            segments and subsegments are actually the same.
        """

        if not os.path.exists(input_file_dir):
            raise ValueError(input_file_dir + ' does not exist!')

        try:
            vec_files = []
            seg_name = None
            org_subseg_name = None

            for root, dirs, files in os.walk(input_file_dir):
                for file in files:
                    if file.endswith('.txt'):
                        if (
                            'Vectors' in file\
                            and identify_flag in file\
                            and seg2vec_method in file
                        ):
                            vec_files.append(os.path.join(root, file))
                        if 'SegmentNames' in file and '150bp' in file:
                            seg_name = os.path.join(root, file)
                        if 'random' in file and 'OriginalSegmentNames' in file:
                            org_subseg_name = os.path.join(root, file)

            try:
                assert len(vec_files) == 2
                for file in vec_files:
                    # 容易混淆, 要加一个特殊符号
                    if '/SegmentVectors' in file:
                        seg_vec = file
                    if '/SubSegmentVectors' in file:
                        subseg_vec = file
                assert seg_vec is not None and subseg_vec is not None
            except ValueError:
                print('files fail to match')

        except IOError:
            print('files not found or fail to read')

        return seg_name, seg_vec, org_subseg_name, subseg_vec

    def get_vecs_to_compare(self):
        """ 4*3套向量 """

        vec_groups = list()  # ordered by self.params
        for param in self.params:

            vec_group = list()  # [[seg_name, seg_vec, org_subseg_name, subseg_vec of <AVG>], [...<TFIDF>], [...<SIF>]]

            for seg2vec_method in self.seg2vec_method:
                seg_name, seg_vec, org_subseg_name, subseg_vec = self.ptr_to_file(
                    identify_flag='-q' + str(param[1]), # 以q识别即可，p均为1
                    seg2vec_method=seg2vec_method,
                    input_file_dir=f"{self.faiss_vec_input_dir}",
                )
                vec_group.append([seg_name, seg_vec, org_subseg_name, subseg_vec])

            vec_groups.append(vec_group)

        return vec_groups

    def multi_test(self):
        """ 设置关键文件路径 """
        # 4套 P Q 参数 -> 4套 seg+subseg 向量
        seg_and_subseg_vecs = self.get_vecs_to_compare()
        index_method = 'HNSW'

        # set a log for recording the result
        logger = Tee(self.faiss_log)
        sys.stdout = logger

        for i in range(len(seg_and_subseg_vecs)):  # 第 i 套 P.Q 参数
            for j in range(len(self.seg2vec_method)):  # 第 i 套 P.Q参数 的 AVG/TFIDF/SIF方法

                faiss_idx_path = self.faiss_idx + f"VecGroupOfParam{i}-{self.seg2vec_method[j]}-{index_method}"

                seg_name_path = seg_and_subseg_vecs[i][j][0]
                seg_vec_path = seg_and_subseg_vecs[i][j][1]
                subseg_name_path = seg_and_subseg_vecs[i][j][2]
                subseg_vec_path = seg_and_subseg_vecs[i][j][3]

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


def main():
    clf = ParamsExploration()
    # clf.kmer_embeddings()
    # clf.seg_and_subseg_embeddings()
    clf.multi_test()


if __name__ == '__main__':
    main()
