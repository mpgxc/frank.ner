import numpy as np
from time import sleep
from tqdm import tqdm


def build_matrix_embeddings(path, num_tokens, embedding_dim, word_index):
    """
        Função para carregar arquivos pre-treinados em memória
    """

    hits, misses = 0, 0
    embeddings_index = {}

    print('Loading file...')

    sleep(0.5)

    for line in tqdm(open(path, encoding='utf-8')):
        word, coefs = line.split(maxsplit=1)
        embeddings_index[word] = np.fromstring(coefs, "f", sep=" ")

    print("Encontrado %s Word Vectors." % len(embeddings_index))

    sleep(0.5)

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))

    for word, i in tqdm(word_index.items()):
        if i >= num_tokens:
            continue
        try:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                embedding_vector = embeddings_index.get(str(word).lower())
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                    hits += 1
                else:
                    embedding_vector = embeddings_index.get(str(word).upper())
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector
                        hits += 1
                misses += 1
        except:
            embedding_matrix[i] = embeddings_index.get('UNK')

    print("Convertidos: %d Tokens | Perdidos: %d Tokens" % (hits, misses))

    return embedding_matrix


def show_true_pred(index, true_tag, pred_tag, ner_source):
    """
        Função para visualizar Tokens com a classe verdadeira e a prevista
        Use:
            show_true_pred(12, y_true, y_pred, ContextNER)
    """
    word_parser_idx = ner_source.idx2word

    print("{:25}   {:10}   {}".format("Word", "True", "Pred"))
    print("=" * 50)

    try:
        TEST = ner_source.X_array[index]
    except:
        TEST = ner_source.X_array_test[index]

    for word, true, pred in zip([word_parser_idx[x] for x in TEST],
                                true_tag[index],
                                pred_tag[index]):
        if pred != 'PAD':
            print("{:25}   {:10}    {}".format(word, true, pred))


def reshape_y_tags(arr):
    """
        Modifica o shape de um array numpy 
        Ex:
            (100, 32, 10) >> (100, 32)
        Use:
            y_test = reshape_y_tags(ContextNER_obj.y_array)
    """
    return np.argmax(arr, -1)
