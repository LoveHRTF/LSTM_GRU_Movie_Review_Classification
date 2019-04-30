import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.models import load_model

class sentence_model:

    """ Class Object for NLP Model """

    def __init__(self, bs=1024, ep=20, sentence_length=150, dataset='movie_data.csv', test_size=0.2, model_type="lstm", embedding_dim=100):

        """ Parameters """
        self.bs = bs
        self.ep = ep
        self.sentence_length = sentence_length
        self.test_size = test_size
        self.embedding_dim = embedding_dim

        """ Train Specific """
        self.dataset = dataset
        self.model_type = model_type

        """ Others """
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X = None
        self.y = None

        print("Parameters: ")
        print("")
        print("Batch Size: ", self.bs)
        print("Training Epoch: ", self.ep)
        print("Sentence Length: ", self.sentence_length)
        print("Model Selected: ", self.model_type)
        print("")


    def load_data(self):

        # Load Data from csv
        df = pd.DataFrame()
        df = pd.read_csv('movie_data.csv', encoding='utf-8')
        df = shuffle(df)

        # Split X and y
        self.X = df.loc[:, 'review'].values
        self.y = df.loc[:, 'sentiment'].values
            
        return self

    
    def data_preprocess(self):

        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, self.y, test_size=self.test_size, shuffle=True)

        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)

        print("Shape for training data is:")
        print("X: ", X.shape)
        print("y: ", y.shape)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X = X
        self.y = y

        return self

    
    def data_tokenize(self):

        # New tokenizer object
        tokenizer_obj = Tokenizer()
        tokenizer_obj.fit_on_texts(self.X) 

        # Define vocabulary size
        vocab_size = len(tokenizer_obj.word_index) + 1
        print("Size of Vocab is: ", vocab_size)

        # Tokenize
        X_train_tokens =  tokenizer_obj.texts_to_sequences(self.X_train)
        X_test_tokens = tokenizer_obj.texts_to_sequences(self.X_test)

        # Padding on dataset
        X_train_pad = pad_sequences(X_train_tokens, maxlen=self.sentence_length, padding='post')
        X_test_pad = pad_sequences(X_test_tokens, maxlen=self.sentence_length, padding='post')
        
        self.vocab_size = vocab_size
        self.X_train = X_train_pad
        self.X_test= X_test_pad
        self.tokenizer_obj = tokenizer_obj

        return self


    def model_build(self):

        # Build Model
        model = Sequential()
        model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.sentence_length))
        if self.model_type == "lstm":
            model.add(LSTM(units=32,  dropout=0.2, recurrent_dropout=0.2))
        elif self.model_type == "gru":
            model.add(GRU(units=32,  dropout=0.2, recurrent_dropout=0.2))
        else:
            return ("ERROR => Invalid Model Selection")

        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print('Summary of the built model...')
        print(model.summary())


        self.model = model

        return self

    
    def model_train(self):

        self.model.fit(self.X_train, self.y_train, \
            batch_size=self.bs, epochs=self.ep, 
            validation_data=(self.X_test, self.y_test), 
            verbose=1)

        return self


    def model_save(self):

        if self.model_type == "lstm":
            self.model.save('lstm_movie_model.h5')

        elif self.model_type == "gru":
            self.model.save('gru_movie_model.h5')

        else:
            print("ERROR => Invalid Model Selection")

        pass
    
    
    def model_test_score(self):

        score, acc = self.model.evaluate(self.X_test, self.y_test, batch_size=self.bs)

        print('Test score:', score)
        print('Test accuracy:', acc)

        print("Accuracy: {0:.2%}".format(acc))

        pass

    def model_load(self, path):
        self.model = load_model(path)
        print(self.model.summary())

        return self


    def model_test_sample(self):

        test_sample_1 = "This movie is fantastic! I really like it because it is so good!"
        test_sample_2 = "Good movie!"
        test_sample_3 = "Maybe I like this movie."
        test_sample_4 = "Not to my taste, will skip and watch another movie"
        test_sample_5 = "if you like action, then this movie might be good for you."
        test_sample_6 = "Bad movie!"
        test_sample_7 = "Not a good movie!"
        test_sample_8 = "This movie really sucks! Can I get my money back please?"
        test_samples = [test_sample_1, test_sample_2, test_sample_3, test_sample_4, test_sample_5, test_sample_6, test_sample_7, test_sample_8]

        test_samples_tokens = self.tokenizer_obj.texts_to_sequences(test_samples)
        test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=self.sentence_length)

        # Predice
        print(self.model.predict(x=test_samples_tokens_pad))

        #let us check how the model predicts
        classes = self.model.predict(self.X_test[:20], batch_size=self.bs)
        for i in range (0,20):
            if(classes[i] > 0.5 and self.y_test[i] == 1 or (classes[i] <= 0.5 and self.y_test[i] == 0)):
                print( classes[i], self.y_test[i], " Right prdiction")
            else :
                print( classes[i], self.y_test[i], " Wrong prdiction")

        pass


    def model_use(self, prediction_sentence):

        print("Input Sentence is: ", prediction_sentence)
        
        test_samples_tokens = self.tokenizer_obj.texts_to_sequences(prediction_sentence)
        test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=self.sentence_length)

        # Predice
        rating_score = self.model.predict(x=test_samples_tokens_pad)
        print("Prediction score is: ", rating_score)

        return rating_score


""" Wrapper functions for basic operation """
def wrapper_initialize():

    nlp_model = sentence_model()

    nlp_model.load_data()
    nlp_model.data_preprocess()
    nlp_model.data_tokenize()

    return nlp_model

def wrapper_train(initialized_model):

    nlp_model = initialized_model
    nlp_model.model_build()
    nlp_model.model_train()
    nlp_model.model_save()

    return nlp_model

def wrapper_test(trained_model):

    nlp_model = trained_model
    nlp_model.model_test_score()
    nlp_model.model_test_sample()

    pass


def wrapper_use(initialized_model, movie_name, review_string, pre_trained_model_path):

    nlp_model = initialized_model
    nlp_model.model_load(pre_trained_model_path)
    rate = nlp_model.model_use(review_string)

    return (movie_name, rate)


""" Wrapper functions that ready to use"""
def train():

    """ Initilize Model """
    model_ini = wrapper_initialize()

    """ Train Model """
    model_trained = wrapper_train(model_ini)

    """ Test Model"""
    wrapper_test(model_trained)

    print("")
    print("Model Saved to $Root")

    pass


def predict(movie,review,model):

    """ Initilize Model """
    model_ini = wrapper_initialize()

    """ Prediction """
    result = wrapper_use(model_ini, movie, review, model)

    return result


if __name__ == "__main__":

    train()