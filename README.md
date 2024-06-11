SeqGraph2Vec
====================================
SeqGraph2Vec is an open-source library to train fast and high-quality k-mer embedding from the k-mer graph. 
Within the k-mer embedding, we can easily compute the DNA sequence embedding. In our paper, the DNA sequence embedding is an average of the k-mer embedding.

See more details in the paper: [Kmer-Node2Vec: a Fast and Efficient Method for Kmer Embedding from the Kmer Co-occurrence Graph, with Applications to DNA Sequences](https://www.biorxiv.org/content/10.1101/2022.08.30.505832v3)

------------------------------------

### Requirements
The codebase is implemented in Python 3.8 package versions. 

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


### Datasets
<p align="justify">
The code takes FASTA format files with file extension of **.fna**. Note that all training FASTA format files should be under the same input directory. A sample FASTA format file is included in the  `data_dir/input/` directory. </p>
<p align="justify">
Training the model is handled by the `src/cli.py` script which provides the following command line arguments.</p>

#### Input and output options
```
  --input-seqs-dir   STR   Sequence files directory.   Default is `data_dir/input/`.
  --output           STR   K-mer embedding path.       Default is `data_dir/input/kmer-embedding.txt`.
  --edge-list-file   STR   Edge file path.             Default is `data_dir/input/edge-list-file.edg`.
```
#### Random walk options
```
  --window-size      INT    Skip-gram window size.        Default is 10.
  --walk-number      INT    Number of walks per node.     Default is 40.
  --walk-length      INT    Number of nodes in walk.      Default is 150.
  --P                FLOAT  Return parameter.             Default is 1.0.
  --Q                FLOAT  In-out parameter.             Default is 0.001.
```
#### Factorization options
```
  --dimensions       INT      Number of dimensions.      Default is 128
  --min-count        INT      Minimal count.             Default is 1
  --workers          INT      Number of cores.           Default is 4.
  --epochs           INT      Number of epochs.          Default is 1.
```

#### Feature creation options

```
  --mer              INT      Length of a sliding window to fragment Mer.         Default is 8.
```

### Examples
<p align="justify">

(1). Only learn the k-mer embedding with Kmer-Node2Vec algorithm. The first example trains an embedding based on the default dataset with standard hyperparameter settings. The script saves the embedding at the default path.</p>
```
python3 src/cli.py
```

Using a custom length of mer for the embedding.
```
python3 src/cli.py --mer 8
```
Using a custom factorization dimension for the embedding.
```
python3 src/cli.py --dimensions 64
```
Using second-order ranom walks for sampling.
```
python3 src/cli.py --P 1 --Q 0.001
```

(2). Evaluate the k-mer embedding in tasks of sequence retrieval and classification. For sequence retrieval task, jump to examples/retrieval_task.py to adjust hyperparameter settings and run the task. For sequence classification task, jump to examples/classification_task.py to adjust hyperparameter settings and run the task. In sequence classification task, we have to obtain pre-trained DNA sequence embedding (an average of k-mer embedding in our paper) in advance, and we offer an implementation of computing DNA sequence embedding within k-mer embedding in examples/retrieval_task.py.

--------------------------------------------------------------------------------
