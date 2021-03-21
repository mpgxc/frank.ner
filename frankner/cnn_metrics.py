import numpy as np
from keras.callbacks import Callback
from seqeval.metrics import f1_score
from seqeval.metrics import recall_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import precision_score
from seqeval.metrics import classification_report

class F1Metrics(Callback):

    def __init__(self, id2label, pad_value=0, validation_data=None, digits=4):

        super(F1Metrics, self).__init__()
        
        self.id2label = id2label
        self.pad_value = pad_value
        self.validation_data = validation_data
        self.digits = digits
        self.is_fit = validation_data is None
        
    def convert_idx_to_name(self, y, array_indexes):

        y = [[self.id2label[idx] for idx in row[row_indexes]] for
             row, row_indexes in zip(y, array_indexes)]
        return y

    def predict(self, X, y):

        y_pred = self.model.predict_on_batch([X[0], X[1]])

        # reduce dimension.
        y_true = np.argmax(y, -1)
        y_pred = np.argmax(y_pred, -1)

        non_pad_indexes = [np.nonzero(y_true_row != self.pad_value)[0] for y_true_row in y_true]

        y_true = self.convert_idx_to_name(y_true, non_pad_indexes)
        y_pred = self.convert_idx_to_name(y_pred, non_pad_indexes)

        return y_true, y_pred

    def score(self, y_true, y_pred):

        f_score = f1_score(y_true, y_pred)
        r_score = recall_score(y_true, y_pred)
        p_score = precision_score(y_true, y_pred)
        
        print('NER Métricas > precision_score: {:04.2f}  --  recall_score: {:04.2f}  --  f1_score: {:04.2f}'.format(p_score, r_score, f_score))
        
        return f_score, r_score, p_score

    def on_epoch_end(self, epoch, logs={}):
        
        if self.is_fit:
            self.on_epoch_end_fit(epoch, logs)
        else:
            self.on_epoch_end_fit_generator(epoch, logs)

    def on_epoch_end_fit(self, epoch, logs={}):
        
        X = self.validation_data[0]
        XC = self.validation_data[1]
        y = self.validation_data[2]
        y_true, y_pred = self.predict([X, XC], y)

        f_score,\
        r_score,\
        p_score = self.score(y_true, y_pred)
        
        logs['f1'], logs['recall'], logs['precision'] = f_score, r_score, p_score
        

    def on_epoch_end_fit_generator(self, epoch, logs={}):
        
        y_true = []
        y_pred = []
        
        for X, y in self.validation_data:
            y_true_batch, y_pred_batch = self.predict(X, y)
            y_true.extend(y_true_batch)
            y_pred.extend(y_pred_batch)

        f_score,\
        r_score,\
        p_score = self.score(y_true, y_pred)
        
        logs['f1'], logs['recall'], logs['precision'] = f_score, r_score, p_score
        
