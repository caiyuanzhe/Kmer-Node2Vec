# -*- coding: utf-8 -*-
import numba
import numpy as np
import networkx as nx
from typing import List
from collections import defaultdict
from gensim.models import Word2Vec

from src import walker
from src.generators import extract_kmer


class KMerNode2Vec:
    """ Save a txt file recording all k-mers' related vectors.

    Args:
        p (float) : return parameter, optional (default = 1)
            The value less than 1 encourages returning back to
            previous vertex, and discourage for value grater than 1.
        q (float) : in-out parameter, optional (default = 0.001)
            The value less than 1 encourages walks to
            go "outward", and value greater than 1
            encourage walking within a localized neighborhood.
        dimensions (int) : dimensionality of the word vectors
            (default = 128).
        num_walks (int): number of walks starting from each node
            (default = 10).
        walks_length (int): length of walk
            (default = 80).
        window (int) : Maximum distance between the current and
            predicted k-mer within a sequence (default = 10).
        min_count (int) : Ignores all k-mers with total frequency
            lower than this (default = 1)
        epochs : Number of iterations (epochs) over the corpus
            (default = 1)
        workers (int) :  number of threads to be spawned for
            runing node2vec including walk generation and
            word2vec embedding, optional (default = 4)
        verbose (bool) : Whether or not to display walk generation
            progress (default = True).

    """

    def __init__(
        self,
        p: float = 1.0,
        q: float = 0.001,
        dimensions: int = 128,
        num_walks: int = 40,
        walks_length: int = 150,
        window: int = 10,
        min_count: int = 1,
        epochs: int = 1,
        workers: int = 4,
        verbose: bool = True,
    ):
        self.p = p
        self.q = q
        self.dimensions = dimensions
        self.num_walks = num_walks
        self.walks_length = walks_length
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.verbose = verbose
        self.workers = workers
        self.edge_list_path = None

    def _generate_graph_file(
        self,
        seqs: List[str],
        mer: int,
        path: str,
    ):
        weight_dict = defaultdict(int)
        for seq in seqs:
            k_mers = extract_kmer(seq, mer)
            for i in range(len(k_mers) - 1):
                weight_dict[(k_mers[i], k_mers[i + 1])] += 1
        
        edge_list = [
             (nodes[0], nodes[1], weight) for nodes, weight in weight_dict.items()
        ]

        with open(path, 'w', encoding='utf-8') as edge_list_file:
            for edge_pair in edge_list:
                write_content = str(edge_pair[0]) + '\t' + str(edge_pair[1]) + '\t' + str(edge_pair[2]) + '\n'
                edge_list_file.write(write_content)
        self.edge_list_path = path

    def _read_graph(self, extend: bool = False):
        walker_mode = getattr(walker, 'SparseOTF', None)
        graph = walker_mode(
            p=self.p,
            q=self.q,
            workers=self.workers,
            verbose=self.verbose,
            extend=extend,
            random_state=None
        )
        graph.read_edg(self.edge_list_path, weighted=True, directed=True)
        return graph

    def _simulate_walks(self, graph: nx.classes.graph.Graph) -> walker.SparseOTF:
        return graph.simulate_walks(self.num_walks, self.walks_length)

    def _learn_embeddings(
        self,
        walks: walker.SparseOTF,
        path_to_embeddings_file: str,
    ):
        if self.workers == 0:
            self.workers = numba.config.NUMBA_DEFAULT_NUM_THREADS
        numba.set_num_threads(self.workers)

        model = Word2Vec(
            walks,
            vector_size=self.dimensions,
            window=self.window,
            min_count=self.min_count,
            sg=1,
            workers=self.workers,
            epochs=self.epochs,
        )
        
        output_fp = path_to_embeddings_file
        if output_fp.endswith(".npz"):
            np.savez(output_fp, IDs=model.wv.index_to_key, data=model.wv.vectors)
        else:
            model.wv.save_word2vec_format(output_fp)

    def fit(
        self,
        seqs: List[str],
        mer: int,
        path_to_edg_list_file: str,
        path_to_embeddings_file: str,
    ):
        """ Get embeddings of k-mers fragmented from input sequences.

        Args:
            graph (nx.classes.graph.Graph) : nx.DiGraph() object.
            seqs (List[str]) : input sequences list.
            mer (int) : sliding window length to fragment k-mers.
                slide only a single nucleotide.
            path_to_edg_list_file (str) : path to k-mers' edges list file.
            path_to_embeddings_file (str) : path to k-mers' embeddings file.
        """
        self._generate_graph_file(seqs, mer, path_to_edg_list_file)
        graph = self._read_graph()
        walks = self._simulate_walks(graph)
        self._learn_embeddings(walks, path_to_embeddings_file)


