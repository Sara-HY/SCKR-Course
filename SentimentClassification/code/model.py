from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping

def LSTMmodel(hidden_state_dim, embeddings, vocabulary_dim, emb_dim, maxlen, input_dropout_rate=0, verbose=False):

   
    model = Sequential()
    model.add(Embedding(vocabulary_dim,
                        emb_dim,
                        input_length=maxlen,
      weights=[embeddings]))
    
    if input_dropout_rate:
        model.add(Dropout(input_dropout_rate))

    model.add(LSTM(hidden_state_dim))

    model.add(Dense(3, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    
    if verbose:
        print(model.summary())
    
    return model


def BidirectionalLSTMmodel(hidden_state_dim, embeddings, vocabulary_dim, emb_dim, maxlen, input_dropout_rate=0, verbose=False):

    model = Sequential()
    model.add(Embedding(vocabulary_dim,
                        emb_dim,
                        input_length=maxlen,
      weights=[embeddings]))
    
    if input_dropout_rate:
        model.add(Dropout(input_dropout_rate))

    model.add(Bidirectional(LSTM(hidden_state_dim)))

    model.add(Dense(3,activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    
    if verbose:
        print(model.summary())
    
    return model