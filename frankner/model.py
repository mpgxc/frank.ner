import keras
from keras import regularizers
from keras.initializers import RandomUniform
from keras.optimizers import Adam
from keras.models import Sequential, Model

from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import concatenate
from keras.layers import MaxPooling1D
from keras.layers import Bidirectional
from keras.layers import TimeDistributed

from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy


def BiLSTM_CNN(isa_crf=True,
               words_weights=None,
               pre_trained=False,
               max_len=None,
               max_len_char=None,
               num_words=None,
               num_tags=None,
               n_chars=None,
               kernel_size=None,
               filter_size=None,
               hiden_layer=128,
               is_trainable=False,
               l2_reg=1e-6,
               learning_rate=0.001,
               dropout=0.5):

    # Char Embbedings
    chars_input = Input(shape=(max_len, max_len_char,), dtype='int32')
    chars_embed = Embedding(input_dim=n_chars,
                            output_dim=max_len_char,
                            embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5))(chars_input)

    chars_embed = TimeDistributed(Dropout(rate=dropout))(chars_embed)

    # Cnn Features
    chars_embed = TimeDistributed(Conv1D(kernel_size=kernel_size,
                                         filters=filter_size,
                                         padding='same',
                                         activation='tanh',
                                         strides=1))(chars_embed)

    chars_embed = TimeDistributed(MaxPooling1D(
        pool_size=max_len_char))(chars_embed)
    chars_embed = TimeDistributed(Flatten())(chars_embed)

    # Word Embbedings
    words_input = Input(shape=(max_len,), dtype='int32')
    words_embed = Embedding(input_dim=words_weights.shape[0],
                            output_dim=words_weights.shape[1],
                            weights=[words_weights],
                            trainable=is_trainable)(words_input)

    x = concatenate([words_embed, chars_embed])

    BiLSTM = Bidirectional(LSTM(units=hiden_layer // 2,
                                return_sequences=True,
                                recurrent_dropout=0.1,
                                kernel_regularizer=regularizers.l2(l2_reg)))(x)

    BiLSTM = TimeDistributed(Dropout(rate=dropout))(BiLSTM)
    BiLSTM = TimeDistributed(Dense(units=hiden_layer,
                                   activation='relu'))(BiLSTM)

    BiLSTM = CRF(units=num_tags)(BiLSTM)

    model = Model(inputs=[words_input, chars_input],
                  outputs=BiLSTM)

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=crf_loss,
                  metrics=[crf_accuracy])
    return model


def BiLSTM(isa_crf=True,
           words_weights=None,
           pre_trained=False,
           max_len=None,
           num_words=None,
           num_tags=None,
           hiden_layer=128,
           is_trainable=False,
           l2_reg=1e-6,
           learning_rate=0.001,
           dropout=0.5):
    """
        Parâmetros:
            isa_crf: isa_crf=True adiciona a camada CRF no topo do modelo, caso isa_crf=False,
            é definido camada Dense com Ativação Softmax para classificação.

            pre_trained: pre_trained=True, adiciona uma layer customizada com pesos de Word
            Embeddings (Glove, FastText)

            is_trainable: is_trainable=True Finetune inicializando os pesos da rede com os Word Embeddings,
            caso is_trainable:False utiliza os pesos dos Word Embeddings para todo treinamento.

            Recomendação!

            Ex:
                model = build_BiLSTM(isa_crf=True,
                                     pre_trained=True,
                                     trainable=True, ...):
    """

    model = Sequential()

    if pre_trained:
        model.add(Embedding(input_dim=words_weights.shape[0],
                            output_dim=words_weights.shape[1],
                            input_length=max_len,
                            weights=[words_weights],
                            trainable=is_trainable))
    else:
        model.add(Embedding(input_dim=num_words,
                            output_dim=max_len,
                            input_length=max_len,
                            trainable=is_trainable))

    model.add(Bidirectional(LSTM(units=hiden_layer // 2,
                                 return_sequences=True,
                                 recurrent_dropout=0.1,
                                 kernel_regularizer=regularizers.l2(l2_reg))))

    model.add(TimeDistributed(Dropout(rate=dropout)))
    # alternative Tanh
    model.add(TimeDistributed(Dense(units=hiden_layer, activation='relu')))

    if isa_crf:

        model.add(CRF(units=num_tags))
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss=crf_loss,
                      metrics=[crf_accuracy])
    else:

        model.add(Dense(units=num_tags, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model
