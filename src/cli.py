# -*- coding: utf-8 -*-
import sys
sys.path.extend(['.', '..'])
import argparse
from prettytable import PrettyTable

from src.generators import parse_seq
from src.kmernode2vec import KMerNode2Vec


class ParameterParser:
    
    def __init__(self, print_params: bool = True):
        self.print_params = print_params
        self.parser = argparse.ArgumentParser(
            description="Run Kmer-Node2Vec."
        )
        self.parsed_args = None

    def parameter_parser(self):
        """ A method to parse up command line parameters.

        Note:
            By default it gives an embedding of (...) dataset.
            The default hyperparameters give a good quality representation.
        """

        self.parser.add_argument(
            '--input-seqs-dir',
            nargs='?',
            default='../data_dir/input/',
            help='Sequence files directory.'
        )

        self.parser.add_argument(
            '--edge-list-file',
            nargs='?',
            default='../data_dir/input/edge-list-file.edg',
            help='Edge file path.'
        )

        self.parser.add_argument(
            '--output',
            nargs='?',
            default='../data_dir/input/kmernode2vec.txt',
            help='Embeddings path.'
        )

        self.parser.add_argument(
            '--mer',
            nargs='?',
            default=8,
            help='Length of a sliding window to fragment mer.'
        )

        self.parser.add_argument(
            '--P',
            type=float,
            default=1.0,
            help='Return hyperparameter. Default is 1.0.'
        )

        self.parser.add_argument(
            '--Q',
            type=float,
            default=0.001,
            help='In-out hyperparameter. Default is 0.001.'
        )

        self.parser.add_argument(
            '--dimensions',
            type=int,
            default=128,
            help='Number of dimensions. Default is 128.'
        )

        self.parser.add_argument(
            '--walk-number',
            type=int,
            default=40,
            help='Number of walks. Default is 40.'
        )

        self.parser.add_argument(
            '--walk-length',
            type=int,
            default=150,
            help='Walk length. Default is 150.'
        )

        self.parser.add_argument(
            '--window-size',
            type=int,
            default=10,
            help='Maximum distance between the current and predicted word within a sentence. Default is 10.'
        )

        self.parser.add_argument(
            '--min-count',
            type=int,
            default=1,
            help='Minimal count. Default is 1.'
        )

        self.parser.add_argument(
            '--workers',
            type=int,
            default=4,
            help='Number of cores. Default is 4.'
        )

        self.parser.add_argument(
            '--epochs',
            type=int,
            default=1,
            help='Number of epochs. Default is 1.'
        )

        if self.print_params is True:
            self._params_printer()

        return self.parser.parse_args()

    def _params_printer(self):
        """ Function to print the logs in a nice table format. """
        parsed_args = vars(self.parser.parse_args())
        table = PrettyTable(["Parameter", "Value"])
        for k, v in parsed_args.items():
            table.add_row([k.replace("_", " ").capitalize(), v])
        print(table)


def main(args):
    clf = KMerNode2Vec(
        p=args.P,
        q=args.Q,
        dimensions=args.dimensions,
        num_walks=args.walk_number,
        walks_length=args.walk_length,
        window=args.window_size,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
    )
    clf.fit(
        seqs=parse_seq([args.input_seqs_dir]),
        mer=args.mer,
        path_to_edg_list_file=args.edge_list_file,
        path_to_embeddings_file=args.output,
    )


if __name__ == "__main__":
    cmd_tool = ParameterParser()
    arguments = cmd_tool.parameter_parser()
    main(arguments)