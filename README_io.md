# Input/Output Specification

## Input/Output Data

Three data format are available: "raw", "sl" (one sentence in a line) and "wl" (one word in a line).
You need to specify a format from them according to tasks and execute modes as below 
(except the case that only one format is supported for the pair of task and mode).

Table 1: available data format for tasks and execute modes

| Task		| Input for train/eval | Input for decode   | Input for interactive | Output for decode |
| :--		| :--		       | :--		    | :--		    | sl[w] or wl[w]    |
| (dual_)seg	| sl[w] or wl[w]       | raw		    | raw		    | sl[wp] or wl[wp]  |
| (dual_)segtag | sl[wp] or wl[wp]     | raw		    | raw		    | sl[wp] or wl[wp]  |
| (dual_)tag	| sl[wp] or wl[wp]     | sl[w] or wl[w]     | sl[w]  		    | sl[wp] or wl[wp]  |
| dep		| wl[wp?h]	       | sl[wp?] or wl[wp?] | sl[wp?]		    | wl[wp?h]	        |
| tdep		| wl[wp?ha]	       | sl[wp?] or wl[wp?] | sl[wp?]		    | wl[wp?ha]	        |
| sematt	| wl[wp?s]	       | sl[wp?] or wl[wp?] | sl[wp?]		    | wl[wp?s]	        |

In the above table:

- "[...]" indicates necessary attributes for corresponding task.
- "w", "p", "h", "a" and "s" respectively indicates word surface, POS of word, index of head word, arc label and semantic attribute label.
- "?" indicates that the preceding attribute is optional.

Input and output data format can be respectively specified by --input_data_format and --output_data_format options with data path options as follows.

      $ python src/seikanlp.py --task seg --execute_mode train --input_data_format sl --train_data path/to/data [--options]
      $ python src/seikanlp.py --task dep --execute_mode decode --input_data_format wl --decode_data path/to/data [--options]

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
each word and its attribute are written with attirbute delimiter symbol "/" .

~~~~
昨日/名詞 は/助詞 雨/名詞 だっ/助動詞 た/助動詞 。/補助記号
今日/名詞 は/助詞 天気/名詞 が/助詞 良い/形容詞 。/補助記号
~~~~

### wl format

A word and its attributes are written for each line with attribute delimiter symbol "\t". 
Empty line indicates sentence separator.
In the below exapmle, attributes for each word includes word surface, POS, 
head word index in the sentence (0 indicates root) and arc label.

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

### Change delimiter symbols

Delimiter symbols for each format is as below.

| Format | Attribute delimiter | Word delimiter | Sentence delimiter  |
| :--    | :--                 | :--            | :--                 |
| raw    | --                  | --             | "\n"                |
| sl     | "/"                 | " " (space)    | "\n"                |
| wl     | "\t"                | "\n"           | "\n\n" (empty line) |

You can change attribute deimiter symbols for input data 
by including the indicator string in the input data as below.

~~~~
#ATTRIBUTE_DELIMITER=_
昨日_名詞_3_obl
は_助詞_1_case
...
~~~~
Also, you can change attribute deimiter symbols for decoded text by the option --output_attribute_delimiter_sl as below.

      $ python src/seikanlp.py --task dep --execute_mode decode --output_data_format sl --output_attribute_delimiter _ --decode_data path/to/data [--options]

### Change attribute columns

You can change column indicies of attributes in input data 
by including the indicator string in the input data as below.
Negative index indicates that the corresponding attribute is not available in the data and 
it must be explicitly written for optional attribute (indicated by "?" in Table 1).

~~~~
#WORD_COLUMN=1
#POS_COLUMN=-1
#HEAD_COLUMN=2
#ARC_COLUMN=3
#SEMATT_COLUMN=-1

昨日	3	obl
は	1	case
雨	0	root
だっ	3	cop
た	3	aux
。	3	punct
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

    $ python src/seikanlp.py --taks seg --execute_mode eval --model_path models/main/YYYYMMDD_eX.XXX.npz [--options]


## Embedding Model

You can train token (word or character) embedding model 
using this toolkit (see README.md) or other tools like Word2Vec or GloVe.
Both binary (.bin) and text (.txt) format are available and 
pre-trained embedding model is used by specifying --unigram_embed_model_path option.


## Dictionary feature

For seg and segtag task, dictionary-based discrete features is available in addition to neural features
by specifying feature template (and external dictionary file) as below.

    $ python src/seikanlp.py --taks seg --execute_mode train --feature_template path/to/template --external_dic_path path/to/dic [--options]

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

    $ python src/seikanlp.py --taks seg --execute_mode train --feature_template path/to/template --external_dic_path path/to/dic [--options]

You can specify "default" for --feature_template option instead of path to template file and then 
default features are extracted.
