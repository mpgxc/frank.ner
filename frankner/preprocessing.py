import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


class ContextNER:

    __X, __y = None, None
    X_array, y_array = None, None

    word2idx, idx2word = None, None
    tag2idx, idx2tag = None, None
    y_array_normal = None

    def __init__(self, df, all_words, max_len=None):

        self.__df = df

        self.all_words = set(all_words)
        self.all_tags = set(df.Tag.values)

        self.sentences = self.__build_sentences()

        self.num_words = len(self.all_words) + 2
        self.num_tags = len(self.all_tags) + 1

        if max_len:
            self.max_len = max_len
        else:
            self.max_len = self._get_maxlen()

        self.__build_Xy()
        self.__build_parsers()
        self.__parser_arrays()

    def _get_maxlen(self):
        return max([len(x) for x in self.sentences]) + 1

    def __build_sentences(self):

        return [x for x in self.__df.groupby('Sentence').apply(
            lambda xdef: [x for x in zip(
                xdef.Word.values,
                xdef.Tag.values
            )]
        )]

    def __build_Xy(self):

        self.__X = [[word for word, __ in value] for value in self.sentences]
        self.__y = [[tag for __, tag in value] for value in self.sentences]

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

    def parser2categorical(self, y_pred, y_true):

        pred_tag = [[self.idx2tag[idx] for idx in row] for row in y_pred]
        y_true_tag = [[self.idx2tag[idx] for idx in row] for row in y_true]

        return pred_tag, y_true_tag

    def __parser_arrays(self):

        tmp_X = [[self.word2idx[index] for index in value]
                 for value in self.__X]
        tmp_y = [[self.tag2idx[index] for index in value]
                 for value in self.__y]

        self.X_array = pad_sequences(maxlen=self.max_len,
                                     sequences=tmp_X,
                                     padding="post",
                                     value=0)

        y_pad = pad_sequences(maxlen=self.max_len,
                              sequences=tmp_y,
                              padding="post",
                              value=0)

        self.y_array_normal = y_pad
        self.y_array = np.array(
            [to_categorical(index, num_classes=self.num_tags) for index in y_pad])
