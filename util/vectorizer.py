# -*- coding: utf-8 -*-
import arrow
import numpy as np
import gensim
from tqdm import tqdm
from typing import List
from pecanpy.wrappers import Timer
from collections import defaultdict

from numba import njit
from numba import types
from numba.typed import Dict
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


class BaseVectorizer():

    def __init__(
        self,
        vecs: gensim.models.keyedvectors.KeyedVectors,
    ):
        self.vecs = vecs
        self.embs = []

    def _map_word2vec(self):
        word2vec = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float32[:],
        )

        for word in self.vecs.key_to_index:
            vector = self.vecs[word]
            word2vec[word] = np.array(vector, dtype=np.float32)

        return word2vec

    @Timer('to save embeddings')
    def save_embs_format(self, output_dir: str, file_name: str):
        if len(self.embs) > 0:
            """
            out_file = '{}{}-{}-{}d-{}.{}'.format(
                output_dir if output_dir.endswith('/') else output_dir + '/',
                file_name,
                arrow.utcnow().format('YYYYMMDD-HHmm'),
                len(self.vecs.vectors[0]),  # dimensions
                self.__class__.__name__,
                'txt',
            )
            """
            out_file = '{}{}.{}'.format(
                output_dir if output_dir.endswith('/') else output_dir + '/',
                file_name,
                'txt',
            )
            with open(out_file, 'w'):
                np.savetxt(out_file, self.embs)
        else:
            raise ValueError('Fail to save embeddings')


class SeqVectorizer(BaseVectorizer):

    def __init__(self, vecs):
        BaseVectorizer.__init__(self, vecs)

    @Timer('to get sequence-level embeddings')
    def train(self, sentences: List[str], vector_size: int = 128, mode='hier_pool'):
        """ mode:'mean_pool', 'max_pool', 'hier_pool' """
        word2vec = BaseVectorizer._map_word2vec(self)
        dimensions = len(self.vecs.vectors[0])

        # averaging vectors
        @njit(fastmath=True, nogil=True)
        def mean_pool(sentence, word2vec_diz, vector_size=vector_size):
            embedding = np.zeros((vector_size,), dtype=np.float32)
            bow = sentence.split()
            for word in bow:
                embedding += word2vec_diz[word]
                # print(len(word2vec_diz[word]))
                # print(word2vec_diz[word][0])
            return embedding / float(len(bow))

        # max pool vectors
        @njit(fastmath=True, nogil=True)
        def max_pool(sentence, word2vec_diz, vector_size=vector_size):
            embedding = np.zeros((vector_size,), dtype=np.float32)
            bow = sentence.split()
            for word in bow:
                word_vec = word2vec_diz[word]
                for bit in range(0, vector_size+1):
                    if word_vec[bit] > embedding[bit]:
                        embedding[bit] = word_vec[bit]
            return embedding

        # max pool vectors
        @njit(fastmath=True, nogil=True)
        def hier_pool(sentence, word2vec_diz, window_size=2, vector_size=vector_size):
            mean_pool_embeddings = list()
            bow = sentence.split()

            # local mean pool at a sliding window
            idx = 0
            while (idx < len(bow)):
                tmp_embedding = np.zeros((vector_size,), dtype=np.float32)
                for word in bow[idx:idx+window_size]:
                    tmp_embedding += word2vec_diz[word]
                tmp_embedding /= window_size
                mean_pool_embeddings.append(tmp_embedding)
                idx += window_size - 1

            # global max pool across sliding windows
            embedding = np.zeros((vector_size,), dtype=np.float32)
            for emb in mean_pool_embeddings:
                for bit in range(0, vector_size+1):
                    if emb[bit] > embedding[bit]:
                        embedding[bit] = emb[bit]
            return embedding

        for i in tqdm(range(len(sentences))):
            if mode == 'mean_pool':
                emb = mean_pool(sentences[i], word2vec, vector_size=dimensions)
            elif mode == 'max_pool':
                emb = max_pool(sentences[i], word2vec, vector_size=dimensions)
            else:  # hier_pool
                emb = hier_pool(sentences[i], word2vec, window_size=2, vector_size=dimensions)
            self.embs.append(emb)

        self.embs = np.array(self.embs)

