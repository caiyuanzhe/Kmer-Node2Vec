# -*- coding: utf-8 -*-
import arrow
import numpy as np
import gensim
from tqdm import tqdm
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from pecanpy.wrappers import Timer
from collections import defaultdict
from fse.models import SIF as FAST_SIF
from fse import SplitIndexedList
from fse.models.average import FAST_VERSION

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
        for word, vector in zip(self.vecs.vocab, self.vecs.vectors):
            word2vec[word] = np.array(vector, dtype=np.float32)
        return word2vec

    @Timer('to save embeddings')
    def save_embs_format(self, output_dir: str, file_name: str):
        if len(self.embs) > 0:
            out_file = '{}{}-{}-{}d-k{}-{}.{}'.format(
                output_dir if output_dir.endswith('/') else output_dir + '/',
                file_name,
                arrow.utcnow().format('YYYYMMDD-HHmm'),
                len(self.vecs.vectors[0]),  # dimensions
                len(self.vecs.index2word[0]),  # kmer
                self.__class__.__name__,
                'txt',
            )
            with open(out_file, 'w'):
                np.savetxt(out_file, self.embs)
        else:
            raise ValueError('Fail to save embeddings')


class SIF(BaseVectorizer):

    def __init__(
        self,
        vecs: gensim.models.keyedvectors.KeyedVectors,
        cache_size_gb: int = 1.0,
        sv_mapfile_path: str = None,
        wv_mapfile_path: str = None,
    ):
        super(SIF, self).__init__(vecs)
        self.cache_size_gb = cache_size_gb
        self.sv_mapfile_path = sv_mapfile_path
        self.wv_mapfile_path = wv_mapfile_path

    @Timer('to get embeddings with SIF method')
    def train(self, sentences: List[str]):
        self.vecs = self.map_word_freq(sentences)
        self.embs = self.sif_embeddings(sentences)

    def map_word_freq(self, sentences: List[str]):
        word_cnt = defaultdict(int)
        for s in sentences:
            for w in s.split():
                word_cnt[w] += 1
        for word in self.vecs.vocab:
            self.vecs.vocab[word].count = word_cnt[word]
        return self.vecs

    def sif_embeddings(self, sentences: List[str]):
        assert FAST_VERSION >= 1  # ensure cython routines worked correctly
        train = SplitIndexedList(sentences)
        model = FAST_SIF(
            self.vecs,
            workers=1,
            cache_size_gb=self.cache_size_gb,
            sv_mapfile_path=self.sv_mapfile_path,
            wv_mapfile_path=self.wv_mapfile_path,
        )
        model.train(train)

        return np.array([model.sv[i] for i in range(len(train))])


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


class TFIDF(BaseVectorizer):

    def __init__(self, vecs):
        BaseVectorizer.__init__(self, vecs)

    @Timer('to get embeddings with TFIDF method')
    def train(self, sentences: List[str]):
        word2vec = BaseVectorizer._map_word2vec(self)
        dimensions = len(self.vecs.vectors[0])

        # initial TF-IDF model using sklearn
        tfidfvectorizer = TfidfVectorizer(
            analyzer='word', stop_words=None, dtype=np.float32, sublinear_tf=True
        )
        tfidf_matrix = tfidfvectorizer.fit_transform(sentences)
        feature_names = tfidfvectorizer.get_feature_names()

        # prepare a dict for recording words and their tfidf
        tfidf = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float32,
        )

        # averaging vectors
        @njit(fastmath=True, nogil=True)
        def tfidf_embeddings(word2vec_diz, tfidf_diz, vector_size=128):
            embedding = np.zeros((vector_size,), dtype=np.float32)
            for word, tfidf_score in tfidf_diz.items():
                embedding += (word2vec_diz[word] * tfidf_score)
            return embedding / float(len(tfidf_diz))

        # Reference:
        # https://stackoverflow.com/questions/34449127/sklearn-tfidf-transformer-how-to-get-tf-idf-values-of-given-words-in-documen/54588081
        for doc in tqdm(range(tfidf_matrix.shape[0])):
            feature_index = tfidf_matrix[doc, :].nonzero()[1]
            tfidf_scores = zip(
                feature_index, [tfidf_matrix[doc, x] for x in feature_index]
            )

            for (i, s) in tfidf_scores:
                tfidf[feature_names[i].upper()] = s  # word: tfidf-score

            emb = tfidf_embeddings(word2vec, tfidf, dimensions)
            self.embs.append(emb)
            tfidf.clear()

        self.embs = np.array(self.embs)


