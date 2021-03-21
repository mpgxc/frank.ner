from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')


def plot_recall_f1_precision_loss(H):

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    ax[0].grid(True)

    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Metrics')

    ax[0].plot(H.history['loss'], marker='X', lw=1)
    ax[0].plot(H.history['val_loss'], marker='D', lw=1)

    ax[1].plot(H.history['f1'], marker='X', lw=1)
    ax[1].plot(H.history['recall'], marker='D', lw=1)
    ax[1].plot(H.history['precision'], marker='o', lw=1)

    ax[1].legend(['f1', 'recall', 'precision'])

    ax[0].legend(['loss', 'val_loss'])
    ax[0].set_title('loss X val_loss')
    ax[1].set_title('f1 X recall x precision')


def plot_recall_f1_precision(H):

    plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(H.history['f1'], marker='X', lw=1)
    plt.plot(H.history['recall'], marker='D', lw=1)
    plt.plot(H.history['precision'], marker='o', lw=1)

    plt.legend(['f1_score', 'recall_score', 'precision_score'])

    plt.xlabel('Epochs ')
    plt.ylabel('Metrics')

    plt.title('f1_score X recall_score X precision_score')
