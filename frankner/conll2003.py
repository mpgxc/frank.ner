import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


class ContextNER:

    all_words, all_tags = [], []

    __X_train, __y_train = None, None
    __X_valid, __y_valid = None, None
    __X_test, __y_test = None, None

    X_array_train, y_array_train = None, None
    X_array_valid, y_array_valid = None, None
    X_array_test, y_array_test = None, None

    word2idx, idx2word = None, None
    tag2idx, idx2tag = None, None

    def __init__(self,
                 path_train,
                 path_valid,
                 path_test):

        self.__path_train = path_train
        self.__path_valid = path_valid
        self.__path_test = path_test

        self.__load_and_build()
        self.__build_uniques()
        self.__build_parsers()

        self.num_words = len(self.all_words) + 2
        self.num_tags = len(self.all_tags) + 1
        self.max_len = self.__get_maxlen()

        self.X_array_train, self.y_array_train = self.__parser_arrays(
            self.__X_train, self.__y_train)
        self.X_array_valid, self.y_array_valid = self.__parser_arrays(
            self.__X_valid, self.__y_valid)
        self.X_array_test, self.y_array_test = self.__parser_arrays(
            self.__X_test, self.__y_test)

    def __get_maxlen(self):
        return max([len(x) for x in self.__X_train + self.__X_valid + self.__X_test])

    def __load_files(self, filename):

        words, tags = [], []
        sent, labels = [], []

        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if line:
                    word, tag = line.split('\t')
                    words.append(word)
                    tags.append(tag)
                else:
                    sent.append(words)
                    labels.append(tags)
                    words, tags = [], []

        return sent, labels

    def __load_and_build(self):

        self.__X_train, self.__y_train = self.__load_files(self.__path_train)
        self.__X_valid, self.__y_valid = self.__load_files(self.__path_valid)
        self.__X_test, self.__y_test = self.__load_files(self.__path_test)

    def __build_uniques(self):

        tmp_x, tmp_y = [], []

        for idx in self.__X_train + self.__X_valid + self.__X_test:
            for x in idx:
                tmp_x.append(x)

        for idx in self.__y_train + self.__y_valid + self.__y_test:
            for x in idx:
                tmp_y.append(x)

        self.all_words = list(set(tmp_x))
        self.all_tags = list(set(tmp_y))

    def __build_parsers(self):

        self.word2idx = {value: idx + 2 for idx,
                         value in enumerate(self.all_words)}
        self.word2idx["UNK"] = 1  # Palavras Desconhecidas
        self.word2idx["PAD"] = 0  # Padding - Preenchimento

        # Converte um index em Word
        self.idx2word = {idx: value for value, idx in self.word2idx.items()}

        # Converte Tag em ìndice
        self.tag2idx = {value: idx + 1 for idx,
                        value in enumerate(self.all_tags)}
        self.tag2idx["PAD"] = 0  # Padding - Preenchimento

        # Converte index em Tag
        self.idx2tag = {idx: value for value, idx in self.tag2idx.items()}

    def __parser_arrays(self, X, Y):

        tmp_X = [[self.word2idx[index] for index in value] for value in X]
        tmp_y = [[self.tag2idx[index] for index in value] for value in Y]

        x_pad = pad_sequences(maxlen=self.max_len,
                              sequences=tmp_X,
                              padding="post",
                              value=0)

        y_pad = pad_sequences(maxlen=self.max_len,
                              sequences=tmp_y,
                              padding="post",
                              value=0)

        return x_pad, np.array([to_categorical(index, num_classes=self.num_tags) for index in y_pad])

    def parser2categorical(self, y_pred, y_true):

        pred_tag = [[self.idx2tag[idx] for idx in row] for row in y_pred]
        y_true_tag = [[self.idx2tag[idx] for idx in row] for row in y_true]

        return pred_tag, y_true_tag
