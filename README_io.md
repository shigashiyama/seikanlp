# Input/Output Specification

## Input/Output Data

There are three data formats: "raw", "sl" (one sentence in a line) and "wl" (one word in a line).
You need to specify a format from them according to tasks and execute modes (except the case that 
only one format is supported for the pair of task and mode).

### raw format

Unsegmented sentence is written for each line as below.

~~~~
昨日は雨だった。
今日は天気が良い。
~~~~

### sl format

Segmented/tokenized sentence where words are split by space is written for each line as below.

~~~~
昨日 は 雨 だっ た 。
今日 は 天気 が 良い 。
~~~~

If a word has an additional attirbute (e.g. POS tag) other than word surface, 
each word and its attributes are written with attribute delimiter symbol "_" .
(You can switch the attribute deimiter symbol for input data by specifying 
--attribute_delimiter option.)

~~~~
昨日_名詞 は_助詞 雨_名詞 だっ_助動詞 た_助動詞 。_補助記号
今日_名詞 は_助詞 天気_名詞 が_助詞 良い_形容詞 。_補助記号
~~~~

### wl format

A word and its attributes are written for each line with attribute delimiter symbol "\t". 
Empty line indicates sentence separator. In the below exapmle, attributes for each word 
includes word surface, POS, head word index in the sentence (0 indicates root) and arc label.

~~~~
昨日	名詞	3	obl
は	助詞	1	case
雨	名詞	0	root
だっ	助動詞	3	cop
た	助動詞	3	aux
。	補助記号	3	punct

今日	名詞	5	obl
は	助動詞	1	case
天気	名詞	5	nsubj
が	助動詞	3	case
良い	形容詞	0	root
。	補助記号	5	punct
~~~~

### Specify feature columns

You need to specify column indicies of features in input data by specifying options as below.

~~~~
$ python src/seika_parser.py --task tdep --execute_mode train --train_data path/to/data --devel_data path/to/data --token_column_index 0 --head_column_index 2 --arc_column_index 3 --attribute_column_indexes 1
~~~~


## Model

During train mode, following the three model files for a trained model are saved in "models/main" directory.

- {YYYYMMDD}_e{X.XXX}.npz
    - Serialized data of (extension of) chainer.link.Chain object that define model structure.  
      "X.XXX" indicates the corresponding real number of epochs.
- {YYYYMMDD}.s2i
    - Serialized data of seikanlp.dictionary.Dictionary object   
      that manages string to index table of words and attributes.
- {YYYYMMDD}.hyp
    - Text file that lists model hyperparameter values.

To reuse the trained model for re-training, evaluation or decoding, the corresponding s2i and hyp files are 
automatically loaded if you specify path of npz file by --model_path/-m option as below.


~~~~
$ python src/seika_tagger.py --taks seg --execute_mode eval --model_path models/main/YYYYMMDD_eX.XXX.npz [--options]
~~~~


## Embedding Model

You can train token (word or character) embedding model 
using this toolkit (src/tools/train_embedding_model.py) or other tools like Word2Vec or GloVe.
Both binary (.bin) and text (.txt) format are available and 
pre-trained embedding model is used by specifying --unigram_embed_model_path option.


## Dictionary feature

For seg and segtag task, dictionary-based discrete features is available in addition to neural features
by specifying feature template (and external dictionary file) as below.

~~~~
$ python src/seika_tagger.py --task seg --execute_mode train --feature_template path/to/template --external_dic_path path/to/dic [--options]
~~~~

Then, vocabulary is build from words in training data and external word dictionary (if specified), 
and dictionary features are extracted.


### External word dictionary

Word dictionary is a text file listing a word in a line as below.

~~~~
日本
日本語
日本人
...
~~~~

### Feature template

This toolkit contains sample template file models/ftemp/default.tmp

~~~~
$ python src/seikanlp.py --taks seg --execute_mode train --feature_template path/to/template --external_dic_path path/to/dic [--options]
~~~~

You can specify "default" for --feature_template option instead of path to template file and then 
default features are extracted.
