import tensorflow as tf
import pandas as pd

from utils import *
from model import *

if __name__ == '__main__':
    # train_x, train_y = read_data('./data/twitter-2016train-A.txt')
    # valid_x, valid_y = read_data('./data/twitter-2016dev-A.txt')
    # test_x, test_y = read_data('./data/twitter-2016test-A.txt')

    # # print(train_x[0], type(train_y[0]))
    # emb_dim = 300

    # words2ids, embeddings = load_embeddings("data/glove.6B.300d.txt", emb_dim)

    # maxlen = max([len(x) for x in train_x] + [len(x) for x in valid_x] + [len(x) for x in test_x])
    # # maxlen = 10

    # train_x, train_y = transform(train_x, train_y, words2ids, maxlen)
    # valid_x, valid_y = transform(valid_x, valid_y, words2ids, maxlen)
    # test_x, test_y = transform(test_x, test_y, words2ids, maxlen)

    # names = ["LSTM", "BidirectionalLSTM"]
    # models = [LSTMmodel(100, embeddings, embeddings.shape[0], embeddings.shape[1], maxlen), 
    #           BidirectionalLSTMmodel(100, embeddings, embeddings.shape[0], embeddings.shape[1], maxlen)]

    # results = []

    # for model, name in zip(models,names):

    #     print("Model : ",name)
        
    #     print(model.summary())
        
    #     # early_stopping = EarlyStopping(patience=2,monitor="val_loss")

    #     model.fit(train_x, train_y,
    #               batch_size=4,
    #               epochs = 50,
    #               # callbacks=[early_stopping],
    #               validation_data=(valid_x, valid_y))

    #     results.append((name, model.evaluate(test_x, test_y, verbose=0)[1]))

    # results = pd.DataFrame(results)
    
    # print(results)  

    # print(len(valid_accuracy))
    # # dispaly_results("LSTM", train_loss, train_accuracy, valid_loss, valid_accuracy, 1) 

    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []

    pfile = open('run.log', 'r')
    data = pfile.read().split('\n')
    for line in data:
        index1 = line.find("loss: ")
        index2 = line.find("categorical_accuracy: ")
        index3 = line.find("val_loss: ")
        index4 = line.find("val_categorical_accuracy: ")
        if index1 != -1 and index2 != -1 and index3 != -1 and index4 != -1 :
            # print(line[index1 + 6: index1 + 12], line[index2 + 22: index2 + 28], line[index3 + 10: index3 + 16], line[index4 + 26: index4 + 32])
            train_loss.append(float(line[index1 + 6: index1 + 12]))
            train_accuracy.append(float(line[index2 + 22: index2 + 28]))
            valid_loss.append(float(line[index3 + 10: index3 + 16]))
            valid_accuracy.append(float(line[index4 + 26: index4 + 32]))


    # print(len(train_loss), '\n', train_loss, '\n', train_accuracy, '\n', valid_loss, '\n', valid_accuracy)
    dispaly_results("LSTM", train_loss[0:50], train_accuracy[0:50], valid_loss[0:50], valid_accuracy[0:50], 1)
    dispaly_results("BidirectionalLSTM", train_loss[50:], train_accuracy[50:], valid_loss[50:], valid_accuracy[50:], 1)







