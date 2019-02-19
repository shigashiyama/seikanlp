# SeikaNLP (version 0.0.2)

SeikaNLP is a Natural Language Processing Toolkit including word segmenter, general sequence tagger and dependency parser.


## Requirements

- Python 3 (tested with 3.6.0)
- Chainer v3 (tested with 3.2.0)
- CuPy (tested with 2.2.0)
- NumPy (tested with 1.13.3)
- gensim 3.6.0

If you use GPU, the following softwares are also required:

- CUDA (tested with 8.0)
- cuDNN (tested with 5.1.5)


## Installation

Clone/download the git repository of this software.


## Files and Directories

~~~~
+-- data             ... directory to place input data
+-- log              ... directory to export log files
+-- models           ... directory to export/place model files
|  +-- main          ... directory to export model files
|  +-- embed         ... directory to export/place embedding model files
|  +-- ftemp         ... directory to place feature template files
+-- src              ... source code directory
~~~~


## Available Tasks

### Sequence labeling by neural network model

- Word segmentation
    - Given a sequence of characters (sentence), the model segments it into words  
      by predicting a sequences of segmentation labels ({B,I,E,S}).  
      
      $ python src/seika_tagger.py --task/-t seg [--options]

- Joint word segmentation and word-level sequence labeling (typically POS tagging)
    - Given a sequence of characters (sentence), the model segments it into words  
      and assign word-level label (e.g. POS) to each word  
      by predicting joint label ({B,I,E,S}-{X1,X2,...,Xm}).

      $ python src/seika_tagger.py --task/-t segtag [--options]

- Sequence labeling
    - Given a sequence of tokens (sentence), the model assigns a token-level label 
      to each token by predicting a sequence of labels.  
      
      $ python src/seika_tagger.py --task/-t tag [--options]      


### Dependency parsing by neural network model

- (Untyped) dependency parsing
    - Given a sequence of tokens (sentence), the model assigns a head (parent) to each token.
    
      $ python src/seika_parser.py --task/-t dep [--options]

- Typed dependency parsing
    - Given a sequence of tokens (sentence), the model assigns a head (parent) to each token
      and assings a label to each arc (pair of a child and its parent).

      $ python src/seika_parser.py --task/-t tdep [--options]


### Semantic attribute annotation

- Token attribute annotation by rule-based model
    - Given a sequence of tokens (sentence), the model assigns a token-level label (attribute)  
      to each token by outputting a most frequent occurred label of each token.

      $ python src/seika_attribute_annotator.py [--options]

Descriptions of options are shown by executing src/seikanlp.py with --help/-h option.


### Training word embedding model

This toolkit includes a script to train word embedding model using gensim Word2Vec API.

    $ python src/train_embedding_model.py [--options]


## Input/Output Specification

See README_io.md


## Change Log

- 2019-02-19 version 0.1.0b
  - Fix minor bugs and add sample data
- 2019-02-15 version 0.1.0
  - Release


## License

Copyright (c) 2019 Shohei Higashiyama
Released under the MIT license https://opensource.org/licenses/mit-license.php


## Contact

Shohei Higashiyama
National Institute of Information and Communications Technology (NICT), Kyoto, Japan
shohei.higashiyama [at] nict.go.jp


## Citation

TBA
