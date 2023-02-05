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


class AVG(BaseVectorizer):

    def __init__(self, vecs):
        BaseVectorizer.__init__(self, vecs)

    @Timer('to get embeddings with AVG method')
    def train(self, sentences: List[str]):
        word2vec = BaseVectorizer._map_word2vec(self)
        dimensions = len(self.vecs.vectors[0])

        # averaging vectors
        @njit(fastmath=True, nogil=True)
        def avg_embeddings(sentence, word2vec_diz, vector_size=128):
            embedding = np.zeros((vector_size,), dtype=np.float32)
            bow = sentence.split()
            for word in bow:
                embedding += word2vec_diz[word]
            return embedding / float(len(bow))

        for i in tqdm(range(len(sentences))):
            emb = avg_embeddings(sentences[i], word2vec, dimensions)
            self.embs.append(emb)

        self.embs = np.array(self.embs)

