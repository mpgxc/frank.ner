import keras
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers.merge import Concatenate


class BiLSTM:

    def __init__(self,
                 isa_crf=True,
                 words_weights=None,
                 pre_trained=False,
                 max_len=None,
                 num_words=None,
                 num_tags=None,
                 learning_rate=0.001,
                 dropout=0.5):

        self.__isa_crf = isa_crf
        self.__words_weights = words_weights
        self.__pre_trained = pre_trained
        self.__max_len = max_len
        self.__num_words = num_words
        self.__num_tags = num_tags
        self.__learning_rate = learning_rate
        self.__dropout = dropout

    """
        Parâmetros:
            isa_crf: isa_crf=True adiciona a camada CRF no topo do modelo, caso isa_crf=False,
            é definido camada Dense com Ativação Softmax para classificação.
            
            pre_trained: pre_trained=True, adiciona uma layer customizada com pesos de Word
            Embeddings (Glove, FastText)
            
            Recomendação!
            
            Ex:
                bilstm = build_BiLSTM(isa_crf=True, pre_trained=True):
                model = bilstm.build_model()
    """

    def predict(self):
        pass

    def build_char_model(self):

        from keras_contrib.layers import CRF
        from keras_contrib.losses import crf_loss

        word_ids = Input(batch_shape=(None, None), dtype='int32')

        word_embeddings = Embedding(input_dim=self.__num_words,
                                    output_dim=self.__max_len,
                                    input_length=self.__max_len,
                                    trainable=False)(word_ids)

        word_ids_2 = Input(batch_shape=(None, None), dtype='int32')

        word_embeddings_2 = Embedding(input_dim=self.__num_words,
                                      output_dim=self.__max_len,
                                      input_length=self.__max_len,
                                      trainable=False)(word_ids)

        word_embeddings = Concatenate()([word_embeddings, word_embeddings_2])

        word_embeddings = Dropout(self.__dropout)(word_embeddings)

        x = Bidirectional(LSTM(units=self.__max_len // 2,
                               return_sequences=True,
                               recurrent_dropout=0.1))(word_embeddings)

        x = TimeDistributed(Dense(units=self.__num_tags, activation='relu'))(x)

        model = Model(inputs=[word_ids, word_ids_2],
                      outputs=CRF(units=self.__num_tags)(x))
        model.compile(optimizer=Adam(
            learning_rate=self.__learning_rate), loss=crf_loss)

        return model

    def build_model(self):

        model = Sequential()

        if self.__pre_trained:
            model.add(Embedding(input_dim=self.__words_weights.shape[0],
                                output_dim=self.__words_weights.shape[1],
                                input_length=self.__max_len,
                                weights=[self.__words_weights],
                                trainable=False))
        else:
            model.add(Embedding(input_dim=self.__num_words,
                                output_dim=self.__max_len,
                                input_length=self.__max_len,
                                trainable=False))

        model.add(Bidirectional(LSTM(units=self.__max_len // 2,
                                     return_sequences=True,
                                     recurrent_dropout=0.1)))

        model.add(Dropout(self.__dropout))

        model.add(TimeDistributed(
            Dense(units=self.__num_tags, activation='relu')))

        if self.__isa_crf:

            from keras_contrib.layers import CRF
            from keras_contrib.losses import crf_loss

            model.add(CRF(units=self.__num_tags))
            # , metrics=[crf_viterbi_accuracy, crf_accuracy])
            model.compile(optimizer=Adam(
                learning_rate=self.__learning_rate), loss=crf_loss)

        else:

            model.add(Dense(units=self.__num_tags, activation='softmax'))
            model.compile(optimizer=Adam(learning_rate=self.__learning_rate),
                          loss="categorical_crossentropy")

        return model
