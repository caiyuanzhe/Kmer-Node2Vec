# -*- coding: utf-8 -*-
import sys
sys.path.extend(['.', '..'])
import os
from gensim.models import KeyedVectors

from src.generators import parse_seq
from src.generators import seq2segs
from src.generators import seg2sentence
from src.generators import extract_seg
from util.vectorizer import SIF, AVG, TFIDF


SPECIES = 'bacteria'  # STEP1: target species to train embeddings: 'viral' or 'bacteria' or 'fungi'
FNA_SEQ_DIR = f"../data_dir/input/precision_dataset/{SPECIES}"  # dir to store target species dna sequence file
WORK_DIR = f"../data_dir/output/seg2vec/{SPECIES}/"  # dir to output embeddings for target species

# optional k-mer2vec file
W2V_KMG2VEC_FILE = f"../data_dir/output/{'KMer2Vec-20220204-0644.txt'}"  # p=1, q=0.001 -> DFS
W2V_DNA2VEC_FILE = f"../data_dir/output/{'dna2vec-20220126-0343-k8to8-128d-10c-4531Mbp-sliding-l8S.w2v'}"  # dna2vec
W2V_DEMO_FILE = "../data_dir/output/KMerNode2Vec-demo.txt"

PTR_TO_W2V_FILE = W2V_DEMO_FILE  # STEP2: 'W2V_KMG2VEC_FILE' or 'W2V_DNA2VEC_FILE'
KMER2VEC_METHOD = 'dna2vec'  # STEP3: 'kmg2vec' or 'dna2vec'. keep pace with W2V_KMG2VEC_FILE.

SEG_FILE = f"{WORK_DIR}SegmentNames-150bp.txt"
# randomly choose 1k segments from SEG_FILE and extract subsegments from the 1k segments
EXTRACTED_ORGSEG_FILE = f"{WORK_DIR}random-1k-OriginalSegmentNames-150bp.txt"
EXTRACTED_SUBSEG_FILE = f"{WORK_DIR}random-1k-SubSegmentNames-75bp.txt"


def seg_vectorization():

    def simple_example():
        segs = ['ACGTACGTACGT']
        sentences = seg2sentence(segs, 8)  # k-mer with k == 8
        vecs = KeyedVectors.load_word2vec_format(
            PTR_TO_W2V_FILE,
        )
        clf = SIF(
            vecs,
            cache_size_gb=40,
            # sv_mapfile_path="../data_dir/output/seg2vec/TMP-SIF-sentence-vectors",
            # wv_mapfile_path="../data_dir/output/seg2vec/TMP-SIF-word-vectors",
        )
        clf.train(sentences)
        clf.save_embs_format('../data_dir/output/seg2vec', 'seg_vec_demo')

    def real_example():

        seq2segs(
            parse_seq([FNA_SEQ_DIR]),
            150,
            path_to_segs_file=SEG_FILE,
        )

        with open(SEG_FILE, 'r', encoding='utf-8') as fp:
            segs = [line.split('\n')[0] for line in fp.readlines()]

        sentences = seg2sentence(segs, 8)

        # k-mer vectors
        vecs = KeyedVectors.load_word2vec_format(PTR_TO_W2V_FILE)
        print(f"k-mer vectors: {PTR_TO_W2V_FILE}")

        clfs = [AVG(vecs), TFIDF(vecs), SIF(vecs, cache_size_gb=40)]
        for clf in clfs:
            clf.train(sentences)
            clf.save_embs_format(
                WORK_DIR,
                f"{KMER2VEC_METHOD}-{'SegmentVectors'}"
            )
            print('******')

    simple_example()
    # real_example()
    return


def generate_segs():
    if os.path.isfile(EXTRACTED_SUBSEG_FILE):  # prevent overriding
        return EXTRACTED_SUBSEG_FILE

    extract_seg(
        SEG_FILE, 150, 1000,
        path_to_extracted_orgsegs_file=EXTRACTED_ORGSEG_FILE,
        path_to_extracted_subsegs_file=EXTRACTED_SUBSEG_FILE,
    )
    print(f"segs_file: {SEG_FILE}\n"
          f"orgsegs_file: {EXTRACTED_ORGSEG_FILE}\n"
          f"subsegs_file:{EXTRACTED_SUBSEG_FILE}"
          )
    return SEG_FILE, EXTRACTED_ORGSEG_FILE, EXTRACTED_SUBSEG_FILE


def subseg_vectorization():

    _, _, subsegs_file = generate_segs()

    # load subsegments
    with open(subsegs_file, 'r', encoding='utf-8') as fp:
        subsegs = [line.split('\n')[0] for line in fp.readlines()]

    sentences = seg2sentence(subsegs, 8)

    # k-mer2vec file
    vecs = KeyedVectors.load_word2vec_format(PTR_TO_W2V_FILE)
    print(f"k-mer vectors: {PTR_TO_W2V_FILE}")

    # calculate sentence vectors by averaging word vectors
    clfs = [AVG(vecs), TFIDF(vecs), SIF(vecs, cache_size_gb=42)]
    for clf in clfs:
        clf.train(sentences)
        clf.save_embs_format(
            WORK_DIR,
            f"{KMER2VEC_METHOD}-{'SubSegmentVectors'}"
        )


def main():
    seg_vectorization()
    subseg_vectorization()


if __name__ == '__main__':
    main()
