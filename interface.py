
def input_new_phrase(self, text):
    
    """
        Retorna uma lista contendo o primeiro elemento como as Palavras e o segundo elemento as Tags (Classes das palavras)
        
        >> [['Mateus', 'B-PER'], 
            ['Pinto',  'I-PER'],
            ['Garcia', 'I-PER']]
    """
    
    x_new_tokens = [word_idx[word] for word in text.split()]
    
    pred = self.model.predict(np.array([x_new_tokens]))
    pred = np.argmax(pred, axis=-1)[0]
    
    return [[word_list[w], tags[pred]] for (w, pred) in zip(range(len(x_new)), pred)]
