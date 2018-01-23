# SeikaNLP (version 0.0.2)

SeikaNLP is a Natural Language Processing Toolkit including word segmenter, general sequence tagger and dependency parser.


## Requirements

- Python 3 (tested with 3.6.0)
- Chainer v3 (tested with 3.2.0)
- numpy (tested with 1.13.0)
- gensim 2.3.0 or lator

If you use GPU, the following softwares are also required:

- CUDA (tested with 8.0)
- cuDNN (tested with 5.1.5)


## Installation

Clone/download the git repository of this software.


## Files and Directories

~~~~
+-- data       ... directory to place input data
+-- log        ... directory to export log files
+-- models     ... directory to export/place model files
|  +-- main    ... directory to export model files
|  +-- embed   ... directory to export/place embedding model files
|  +-- ftemp   ... directory to place feature template files
+-- resources  ... directory to place resource files such as dictionary
+-- src        ... source code directory
~~~~


## Available Tasks

### Sequence labeling by neural network model

- Word segmentation
    - Given a sequence of characters (sentence), the model segments it into words  
      by predicting a sequences of segmentation labels ({B,I,E,S}).  
      
      $ python src/seikanlp.py --task/-t seg [--options]

- Joint word segmentation and word-level sequence labeling (typically POS tagging)
    - Given a sequence of characters (sentence), the model segments it into words  
      and assign word-level label (e.g. POS) to each word  
      by predicting joint label ({B,I,E,S}-{X1,X2,...,Xm}).

      $ python src/seikanlp.py --task/-t segtag [--options]

- Sequence labeling
    - Given a sequence of tokens (sentence), the model assigns a token-level label to each token  
      by predicting a sequence of labels.  
      
      $ python src/seikanlp.py --task/-t tag [--options]      


### Dual sequence labeling by neural network model

Conduct two sequence labeling tasks simultaneously using two models (e.g. word segmentation based on short unit and one based on long unit).
Three tasks 'dual_seg', 'dual_segtag' and 'dual_tag' are supported and they can be executed similarly to (single) sequence labeling tasks.


### Dependency parsing by neural network model

- (Untyped) dependency parsing
    - Given a sequence of tokens (sentence), the model assigns a head (parent) to each token.
    
      $ python src/seikanlp.py --task/-t dep [--options]

- Typed dependency parsing
    - Given a sequence of tokens (sentence), the model assigns a head (parent) to each token
      and assings a label to each arc (pair of a child and its parent).

      $ python src/seikanlp.py --task/-t tdep [--options]


### Semantic attribute annotation

- Token attribute annotation by rule-based model
    - Given a sequence of tokens (sentence), the model assigns a token-level label (attribute)  
      to each token by outputting a most frequent occurred label of each token.

      $ python src/seikanlp.py --task/-t sematt [--options]

Descriptions of options are shown by executing src/seikanlp.py with --help/-h option.
See also sample.sh to confirm typical usages.


### Training word embedding model

This toolkit includes a sample program to train word embedding model using gensim Word2Vec API.

    $ python src/train_embedding_model.py [--options]


## Input/Output Specification

See README_io.md


## Change Log

- 2018-01-30 version 0.0.2
    - Add dual sequence labeling and semantic attribute annotation module and fix bugs
- 2017-12-27 version 0.0.1b
    - Add dependency parsing module
- 2017-10-30 version 0.0.1
    - Initial release


## Citation

To be announced.


## Contact

Shohei Higashiyama
National Institute of Information and Communications Technology (NICT), Kyoto, Japan
shohei.higashiyama [at] nict.go.jp
