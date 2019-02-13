import pickle
import re
import sys

import constants


class Data(object):
    def __init__(self, inputs=None, outputs=None, featvecs=None):
        self.inputs = inputs      # list of input sequences e.g. chars, words
        self.outputs = outputs    # list of output sequences (label sequences) 
        self.featvecs = featvecs  # feature vectors other than input sequences


class RestorableData(Data):
    def __init__(self, inputs=None, outputs=None, featvecs=None, orgdata=None):
        super().__init__(inputs, outputs, featvecs)
        self.orgdata = orgdata


class DataLoader(object):
    def __init__(self):
        self.token_index = 1
        self.lowercasing = False
        self.normalize_digits = False


    def load_gold_data(self, data_format, path, dic=None, train=True):
        # to be implemented in sub-class
        pass


    def load_decode_data(self, data_format, path, dic=None):
        # to be implemented in sub-class
        pass


    def parse_commandline_input(self, line, dic):
        # to be implemented in sub-class
        pass


    def normalize_input_line(self, line):
        line = re.sub(' +', ' ', line).strip(' \t\n')
        return line
        

    def preprocess_token(self, token):
        if self.lowercasing:
            token = token.lower()
        if self.normalize_digits:
            token = re.sub(r'[0-9]+', constants.NUM_SYMBOL, token)
        return token


    def preprocess_attribute(self, attr, depth, target_labelset):
        if target_labelset and (not attr in self.target_labelset):
            attr = '' # ignore
     
        attr = self.get_sub_attribute(attr, depth)

        return attr


    def get_sub_attribute(self, attr, depth):
        if depth == 1:
            sub_attr = attr.split(constants.SUBATTR_SEPARATOR)[0]
        elif depth > 1:
            sub_attr = constants.SUBATTR_SEPARATOR.join(attr.split(constants.SUBATTR_SEPARATOR)[0:depth])
        else:
            sub_attr = attr
        return sub_attr


    def to_be_registered(self, token, train, freq_tokens=set(), refer_vocab=set()):
        if train:
            if not freq_tokens or token in freq_tokens:
                return True            
        else:
            if token in refer_vocab:
                return True
     
        return False


    def get_frequent_bigrams_SL(self, path, freq_threshold, max_vocab_size=-1):
        attr_delim = constants.WL_ATTR_DELIM
        counter = {}
        ins_cnt = 0
     
        with open(path) as f:
            for line in f:
                line = self.normalize_input_line(line)
                if len(line) < 1:
                    continue
     
                elif line[0] == constants.COMMENT_SYM:
                    continue

                pstr = ''
                entries = line.split()
                for entry in entries:
                    word = entry.split(attr_delim)[0]
                    pstr += word
     
                str_bigrams = create_all_char_ngrams(pstr, 2)
                str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
                for sb in str_bigrams:
                    if sb in counter:
                        counter[sb] += 1
                    else:
                        counter[sb] = 0
     
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        if max_vocab_size > 0:
            counter2 = Counter(counter)
            for pair in counter2.most_common(max_vocab_size):
                del counter[pair[0]]
            
            print('keep {} bigrams from {} bigrams (max vocab size={})'.format(
                len(counter), len(counter2), max_vocab_size), file=sys.stderr)
     
        freq_bigrams = set()
        for word in counter:
            if counter[word] >= freq_threshold:
                freq_bigrams.add(word)
     
        print('keep {} bigrams from {} bigrams (frequency threshold={})'.format(
            len(freq_bigrams), len(counter), freq_threshold), file=sys.stderr)
        
        return freq_bigrams
     
     
    def get_frequent_bigrams_WL(self, path, freq_threshold, max_vocab_size=-1):
        attr_delim = constants.WL_ATTR_DELIM
        counter = {}
        word_clm = self.token_index
        ins_cnt = 0
     
        with open(path) as f:
            pstr = ''
            for line in f:
                line = re.sub(' +', ' ', line).strip(' \t\n')
                if len(line) < 1:
                    str_bigrams = create_all_char_ngrams(pstr, 2)
                    str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
                    for sb in str_bigrams:
                        if sb in counter:
                            counter[sb] += 1
                        else:
                            counter[sb] = 0
     
                    pstr = ''
                    ins_cnt += 1
                    if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                        print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
                    continue
     
                elif line[0] == constants.COMMENT_SYM:
                    continue
                
                array = line.split(attr_delim)
                word = array[word_clm]
                pstr += word
     
            if pstr:
                str_bigrams = create_all_char_ngrams(pstr, 2)
                str_bigrams.append('{}{}'.format(pstr[-1], constants.EOS))
                for sb in str_bigrams:
                    if sb in counter:
                        counter[sb] += 1
                    else:
                        counter[sb] = 0
     
        if max_vocab_size > 0:
            counter2 = Counter(counter)
            counter = {k:v for k,v in counter2.most_common(max_vocab_size)}
            print('keep {} bigrams from {} bigrams (max vocab size={})'.format(
                len(counter), len(counter2), max_vocab_size), file=sys.stderr)
     
        freq_bigrams = set()
        for word in counter:
            if counter[word] >= freq_threshold:
                freq_bigrams.add(word)
     
        print('keep {} bigrams from {} bigrams (frequency threshold={})'.format(
            len(freq_bigrams), len(counter), freq_threshold), file=sys.stderr)
        
        return freq_bigrams
     
     
    def get_frequent_ngrams_SL(self, path, freq_threshold, max_vocab_size=-1, min_word_length=1, max_word_length=4):
        attr_delim = constants.WL_ATTR_DELIM
        counter = {}
        ins_cnt = 0
     
        with open(path) as f:
            for line in f:
                line = re.sub(' +', ' ', line).strip(' \t\n')
                if len(line) < 1:
                    continue
     
                elif line[0] == constants.COMMENT_SYM:
                    continue

                pstr = ''
                entries = line.split()
                for entry in entries:
                    word = entry.split(attr_delim)[0]
                    pstr += word
     
                str_ngrams = create_all_char_ngrams(pstr, 2)
                for n in range(min_word_length, max_word_length+1):
                    str_ngrams = create_all_char_ngrams(pstr, n)
                    for sb in str_ngrams:
                        if sb in counter:
                            counter[sb] += 1
                        else:
                            counter[sb] = 0
     
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        if max_vocab_size > 0:
            counter2 = Counter(counter)
            counter = {k:v for k,v in counter2.most_common(max_vocab_size)}
            
            print('keep {} ngrams from {} ngrams (max vocab size={})'.format(
                len(counter), len(counter2), max_vocab_size), file=sys.stderr)
     
        freq_ngrams = set()
        for word in counter:
            if counter[word] >= freq_threshold:
                freq_ngrams.add(word)
     
        print('keep {} ngrams from {} ngrams (frequency threshold={})'.format(
            len(freq_ngrams), len(counter), freq_threshold), file=sys.stderr)
        
        return freq_ngrams
     
     
    def get_frequent_ngrams_WL(self, path, freq_threshold, max_vocab_size=-1, min_word_length=1, max_word_length=4):
        attr_delim = constants.WL_ATTR_DELIM
        counter = {}
        word_clm = self.token_index
        ins_cnt = 0
     
        with open(path) as f:
            pstr = ''
            for line in f:
                line = re.sub(' +', ' ', line).strip(' \t\n')
                if len(line) < 1:
                    for n in range(min_word_length, max_word_length+1):
                        str_ngrams = create_all_char_ngrams(pstr, n)
                        for sb in str_ngrams:
                            if sb in counter:
                                counter[sb] += 1
                            else:
                                counter[sb] = 0
     
                    pstr = ''
                    ins_cnt += 1
                    if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                        print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
                    continue
     
                elif line[0] == constants.COMMENT_SYM:
                    continue
                
                array = line.split(attr_delim)
                word = array[word_clm]
                pstr += word
     
            if pstr:
                for n in range(min_word_length, max_word_length+1):
                    str_ngrams = create_all_char_ngrams(pstr, n)
                    for sb in str_ngrams:
                        if sb in counter:
                            counter[sb] += 1
                        else:
                            counter[sb] = 0
     
        if max_vocab_size > 0:
            counter2 = Counter(counter)
            counter = {k:v for k,v in counter2.most_common(max_vocab_size)}
            
            print('keep {} ngrams from {} ngrams (max vocab size={})'.format(
                len(counter), len(counter2), max_vocab_size), file=sys.stderr)
     
        freq_ngrams = set()
        for word in counter:
            if counter[word] >= freq_threshold:
                freq_ngrams.add(word)
     
        print('keep {} ngrams from {} ngrams (frequency threshold={})'.format(
            len(freq_ngrams), len(counter), freq_threshold), file=sys.stderr)
        
        return freq_ngrams
     
     
    def get_frequent_tokens_SL(self, path, freq_threshold, max_vocab_size=-1):
        attr_delim = constants.WL_ATTR_DELIM
        counter = {}
        ins_cnt = 0
     
        with open(path) as f:
            for line in f:
                line = re.sub(' +', ' ', line).strip(' \t\n')
                if len(line) < 1:
                    continue
     
                elif line[0] == constants.COMMENT_SYM:
                    continue

                entries = line.split()
                for entry in entries:
                    attrs = entry.split(attr_delim)
                    token = self.preprocess_token(attrs[0])
                    wlen = len(token)
     
                    if token in counter:
                        counter[token] += 1
                    else:
                        counter[token] = 0
     
                ins_cnt += 1
                if ins_cnt % constants.NUM_FOR_REPORTING == 0:
                    print('Read', ins_cnt, 'sentences', file=sys.stderr)
     
        if max_vocab_size > 0:
            counter2 = Counter(counter)
            counter = {k:v for k,v in counter2.most_common(max_vocab_size)}
            
            print('keep {} tokens from {} tokens (max vocab size={})'.format(
                len(counter), len(counter2), max_vocab_size), file=sys.stderr)
     
        freq_tokens = set()
        for token in counter:
            if counter[token] >= freq_threshold:
                freq_tokens.add(token)
     
        print('keep {} tokens from {} tokens (frequency threshold={})'.format(
            len(freq_tokens), len(counter), freq_threshold), file=sys.stderr)
        
        return freq_tokens
     
     
    def get_frequent_tokens_WL(
            self, path, freq_threshold, max_vocab_size=-1, lowercasing=False, normalize_digits=True):
        attr_delim = self.attr_delim if self.attr_delim else constants.WL_ATTR_DELIM
        counter = {}
        word_clm = self.token_index
        ins_cnt = 0
     
        with open(path) as f:
            for line in f:
                line = re.sub(' +', ' ', line).strip(' \n\t')
     
                if len(line) < 1:
                    if ins_cnt > 0 and ins_cnt % constants.NUM_FOR_REPORTING == 0:
                        print('Read', ins_cnt, 'sentences', file=sys.stderr)
                    ins_cnt += 1
                    continue
     
                elif line[0] == constants.COMMENT_SYM:
                    continue
     
                array = line.split(attr_delim)
                token = self.preprocess_token(array[word_clm])
                wlen = len(token)
     
                if token in counter:
                    counter[token] += 1
                else:
                    counter[token] = 0
     
        if max_vocab_size > 0:
            org_counter = counter
            mc = Counter(org_counter).most_common(max_vocab_size)
            print('keep {} tokens from {} tokens (max vocab size={})'.format(
                len(mc), len(org_counter), max_vocab_size), file=sys.stderr)
            counter = {pair[0]:pair[1] for pair in mc}
     
        freq_tokens = set()
        for token in counter:
            if counter[token] >= freq_threshold:
                freq_tokens.add(token)
     
        print('keep {} tokens from {} tokens (frequency threshold={})'.format(
            len(freq_tokens), len(counter), freq_threshold), file=sys.stderr)
        
        return freq_tokens


def load_pickled_data(filename_wo_ext, load_dic=True):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'rb') as f:
        obj = pickle.load(f)
        inputs = obj[0]
        outputs = obj[1]
        featvecs = obj[2]

    return Data(inputs, outputs, featvecs)


def dump_pickled_data(filename_wo_ext, data):
    dump_path = filename_wo_ext + '.pickle'

    with open(dump_path, 'wb') as f:
        obj = (data.inputs, data.outputs, data.featvecs)
        pickle.dump(obj, f)


def create_all_char_ngrams(chars, n):
    seq_len = len(chars)
    if n > seq_len or n <= 0 or seq_len == 0:
        return []
 
    ngrams = []
    for i in range(seq_len-n+1):
        ngrams.append(chars[i:i+n])
    return ngrams
    
 
def create_all_char_ngram_indexes(chars, n):
    seq_len = len(chars)
    if n > seq_len or n <= 0 or seq_len == 0:
        return []
 
    index_pairs = []
    for i in range(seq_len-n+1):
        index_pairs.append((i,i+n))
    return index_pairs


def get_label_BI(index, cate=None):
    prefix = 'B' if index == 0 else 'I'
    suffix = '' if cate is None else '-' + cate
    return prefix + suffix


def get_label_BIES(index, last, cate=None):
    if last == 0:
        prefix = 'S'
    else:
        if index == 0:
            prefix = 'B'
        elif index < last:
            prefix = 'I'
        else:
            prefix = 'E'
    suffix = '' if not cate else '-' + cate
    return '{}{}'.format(prefix, suffix)


# TODO check
def get_labelseq_BIO(seq):
    N = len(seq)
    ret = [constants.O] * N
    for i, elem in enumerate(seq):
        if not elem:
            continue

        prev = seq[i-1] if (i > 0 and seq[i-1]) else None
        if prev != elem:
            ret[i] = '{}-{}'.format(constants.B, elem)
        else:
            ret[i] = '{}-{}'.format(constants.I, elem)

    return ret


def get_labelseq_BIOES(seq):
    N = len(seq)
    ret = [constants.O] * N
    for i, elem in enumerate(seq):
        if not elem:
            continue

        prev = seq[i-1] if (i > 0 and seq[i-1]) else None
        next = seq[i+1] if (i < N-1 and seq[i+1]) else None

        if prev != elem:
            if elem == next:
                ret[i] = '{}-{}'.format(constants.B, elem)
            else:
                ret[i] = '{}-{}'.format(constants.S, elem)
        else:
            if elem == next:
                ret[i] = '{}-{}'.format(constants.I, elem)
            else:
                ret[i] = '{}-{}'.format(constants.E, elem)

    return ret
