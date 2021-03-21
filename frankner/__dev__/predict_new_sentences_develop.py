from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import TweetTokenizer

def parser_string(text, ner_train):
    
    tmp_text_1 = TweetTokenizer().tokenize(text)
    tmp_text_2 = []
    
    for x in tmp_text_1:
        try:
            token_idx = ner_train.word2idx[x] 
        except:
            token_idx = ner_train.word2idx['UNK'] 
            
        tmp_text_2.append(token_idx)

            
    return pad_sequences(maxlen=ner_train.max_len, 
                         sequences=[ np.asarray(tmp_text_2)],
                         padding="post", 
                         value=0)

def cnn_predict_news_ent():

	tweet = "I say burn it all down I and my frind Matthews (metaphorically of course). Abolish Venezuela, China, \
	America, Canada, Australia, Russia, Mexico, every single country on the planet. \
	They’re all illegitimate and only exist because we allow them to"

	tokens_idx = parser_string(tweet)
	tokens_idx_char = get_X_char(tweet, ner_train.max_len, max_len_char)



	preds = model.predict([tokens_idx, tokens_idx_char])
	preds = np.argmax(preds, axis=-1)
	y_pred, _ = ner_train.parser2categorical(preds, preds)


	print("{:10}   {:5}".format("Word", "Pred"))
	print("=" * 50)

	for tok, tag in zip([ner_train.idx2word[x] for x in list(tokens_idx[0])], y_pred[0]):
	    if tag != 'PAD':
	        print("{:10}   {:5}".format(tok, tag))
	        
