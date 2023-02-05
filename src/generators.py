# -*- coding: utf-8 -*-
import os
import re
import time
import random
from random import sample
from Bio import SeqIO
from math import floor
from typing import List
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from gensim import utils


class Timer:
    """Timer for logging runtime of function."""

    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose

    def __call__(self, func):
        """Call timer decorator."""

        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start

            hrs = int(duration // 3600)
            mins = int(duration % 3600 // 60)
            secs = duration % 60
            print(f"Took {hrs:02d}:{mins:02d}:{secs:05.2f} to {self.name}")

            return result

        return wrapper if self.verbose else func


def parse_seq(path_to_input: str):
    """ Return a list containing DNA seqment(s) captured in fna file(s)."""
    seq_files = list()
    for input_file_dir in path_to_input:
        print(input_file_dir)
        for root, dirs, files in os.walk(input_file_dir):
            for file in files:
                if file.endswith('.fna'):
                    seq_files.append(os.path.join(root, file))
    seqs = list()
    for seq_file in seq_files:
        for seq_record in SeqIO.parse(seq_file, 'fasta'):
            seq = re.sub('[^ACGTacgt]+', '', str(seq_record.seq))
            seqs.append(seq.upper())

    print('There are ' + str(len(seqs)) + ' seqs')

    return seqs


def extract_kmer(seq: str, mer: int):
    """ Return a DNA sequence's k-mers. Slide only a single nucleotide """
    return [seq[i:i + mer] for i in range(len(seq) - mer + 1)]


def seg2sentence(segs: List[str], mer: int = 8):
    """ Express a segment in NLP sentence style.
    Note:
        ['segments'] --> ['seg egm gme men ent nts']
    """
    return [' '.join(extract_kmer(seg, mer)) for seg in segs]


def save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
    """ Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility."""
    if not (vocab or vectors):
        raise RuntimeError("no input")
    if total_vec is None:
        total_vec = len(vocab)
    vector_size = vectors.shape[1]
    assert (len(vocab), vector_size) == vectors.shape
    with utils.open(fname, 'wb') as fout:
        print("total_vec:", total_vec, " vector_size:", vector_size)
        fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
        # store in sorted order: most frequent words at the top
        for word, row in vocab.items():
            if binary:
                row = row.astype(REAL)
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))

@Timer('convert DNA seqs to a set of segs')
def seq2segs(
    seqs: List[str],
    step_length: int = 150,
    path_to_segs_file: str = None,
):
    """ Fragment sequences into small segments owning a fixed length.

    Note:
        The function returns a list of segs which looks
        like ['ACGT..', 'TCAG..',  ...]. Each element in
        segs[i] is the sequence fragment having a fixed
        length == step_length
    """

    segs = list()
    for seq in seqs:
        i = 0
        while i <= len(seq) - step_length:
            segs.append(seq[i:i + step_length])
            i += step_length

    if path_to_segs_file is not None:
        with open(path_to_segs_file, 'w', encoding='utf-8') as f:
            for seg in segs:
                f.write(str(seg) + '\n')
    return segs


@Timer('seg2sentence')
def seg2sentence(segs: List[str], mer: int = 8):
    """ Express a segment in NLP sentence style.
    Note:
        ['segments'] --> ['seg egm gme men ent nts']
    """
    return [' '.join(extract_kmer(seg, mer)) for seg in segs]


def check_file_sanity(file: str, line_len: int):
    """ ensure each subsegment's length is half segment length """
    with open(file, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            assert len(line) == line_len + 1  # Typically, line's tail is '/n'


@Timer('to extract subsegments')
def extract_seg(
    path_to_segs_file: str,
    seg_len: int = 150,
    sample_choice: int = 1000,
    path_to_extracted_subsegs_file: str = None,
    path_to_extracted_orgsegs_file: str = None,
):
    """ Randomly extract sub-segments from given segments.

    Args:
        path_to_segs_file (str) : segments file, each line represents
            one segment.
        seg_len (int) : manually input the segment length to validate
            every segment in "path_to_segs_file" share the same length.
        sample_choice (int) : number of sub-segments to random extraction.
        path_to_extracted_subsegs_file (str) : sub-segments file path.
        path_to_extracted_orgsegs_file (str) : sub-segments' original segments file path.
    """
    if os.path.exists(path_to_extracted_subsegs_file):
        raise ValueError('already exists path_to_extracted_subsegs_file')

    elif os.path.exists(path_to_extracted_orgsegs_file):
        raise ValueError('already exists path_to_extracted_orgsegs_file')

    else:
        def random_choose(org_target):
            end_index = floor(len(org_target)/2)
            start_index = random.randrange(0, end_index)
            return org_target[start_index: start_index + end_index]

        with open(path_to_segs_file, 'r', encoding='utf-8') as fp:
            check_file_sanity(path_to_segs_file, seg_len)  # ensure each segment enjoys a fixed length
            orgsegs = [line.split('\n')[0] for line in fp.readlines()]
            # randomly sample numerous segments
            if sample_choice > 0:
                orgsegs = sample(orgsegs, sample_choice)
            subsegs = [random_choose(x) for x in orgsegs]

        if path_to_extracted_orgsegs_file is not None:  # 150bp
            with open(path_to_extracted_orgsegs_file, 'w', encoding='utf-8') as fp:
                for sub in orgsegs:
                    fp.write(str(sub) + '\n')
            check_file_sanity(path_to_extracted_orgsegs_file, seg_len)

        if path_to_extracted_subsegs_file is not None:  # 75bp
            with open(path_to_extracted_subsegs_file, 'w', encoding='utf-8') as fp:
                for sub in subsegs:
                    fp.write(str(sub) + '\n')
            check_file_sanity(path_to_extracted_subsegs_file, floor(seg_len/2))

    return subsegs
