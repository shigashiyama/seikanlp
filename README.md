# SeikaNLP (version 0.0.1)

SeikaNLP is a Natural Language Processing Toolkit including word segmenter, POS tagger and general sequence tagger.


## Requirements

- Python 3 (tested with 3.6.0)
- Chainer v2 (tested with 2.0.0)
- numpy (tested with 1.13.0)
- gensim

If you use GPU, the following softwares are also required:

- CUDA (tested with 8.0)
- cuDNN (tested with 5.1.5)


## Installation

Only need to clone/download the git repository of this software.


## Files and Directories

~~~~
+-- data       ... directory to place input data
+-- log        ... directory to export log files
+-- models     ... directory to export/place model files
|  +-- main    ... directory to export NN model files
|  +-- embed   ... directory to export/place embedding model files
+-- out        ... directory to export output files such as parsed text
+-- resources  ... directory to place resource files such as dictionary
+-- src        ... source code directory
~~~~


## How to Use

~~~~
$ python src/segmenter.py [--options]
~~~~

Descriptions of options are shown by executing with --help or -h option

See Also sample.sh to confirm typical usages.


## Change Log

- 2017-10-30 version 0.0.1
    - Initial release


## Citation

To be announced.


## Contact

Shohei Higashiyama,  
National Institute of Information and Communications Technology (NICT),  
shohei.higashiyama [at] nict.go.jp
