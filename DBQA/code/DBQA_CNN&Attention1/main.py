import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import sys

from data_process import QA_pair, Word2Vec
from CNN_Attention import CNN_Attention
from sklearn import linear_model, svm
from sklearn.externals import joblib


def evaluation(QA_pairs):
    # Calculate MAP and MRR for comparing performance
    MAP, MRR = 0, 0
    for question in QA_pairs.keys():
        p, AP = 0, 0
        MRR_check = False

        QA_pairs[question] = sorted(QA_pairs[question], key=lambda x: x[-1], reverse=True)

        for idx, (answer, label, prob) in enumerate(QA_pairs[question]):
            if label == 1:
                if not MRR_check:
                    MRR += 1 / (idx + 1)
                    MRR_check = True

                p += 1
                AP += p / (idx + 1)

        if p == 0:
            AP = 0
        else:
            AP /= p
        MAP += AP

    num_questions = len(QA_pairs.keys())
    MAP /= num_questions
    MRR /= num_questions

    return MAP, MRR


def train(training_data, testing_data, sentence_length, learning_rate, window_size, l2_reg, training_epochs, batch_size, model_type, classifier, num_classes=2, num_layers=2):
    model = CNN_Attention(s=sentence_length, w=window_size, l2_reg=l2_reg, model_type=model_type, 
            num_features=training_data.num_features, num_classes=num_classes, num_layers=num_layers)

    optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer').minimize(model.cost)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=10)

    MAPs, MRRs = [], []
    with tf.Session() as sess:
        sess.run(init)

        print('=' * 50)
        print('training......')
        for epoch in range(1, training_epochs + 1):
            training_data.reset_index()
            clf_features = []
            avg_cost = 0

            while training_data.is_available():
                batch_questions, batch_answers, batch_labels, batch_features = training_data.next_batch(batch_size=batch_size)

                merged, _, cost, features = sess.run([model.merged, optimizer, model.cost, model.output_features], 
                    feed_dict = {model.x1: batch_questions,
                                    model.x2: batch_answers,
                                    model.y: batch_labels,
                                    model.features: batch_features})
                
                avg_cost += cost / training_data.data_size * len(batch_questions)
                  
                clf_features.append(features)

            print("Epoch ", epoch, ": ./models/" + model_type + '_' + str(epoch), ", Cost:", avg_cost)
           

            saver.save(sess, "./models/" + 'new' + model_type, global_step=epoch)

            for i in range(len(clf_features)):
                if np.isnan(clf_features[i]).any():
                    print("nan clf_features")
                    clf_features[i] = np.nan_to_num(clf_features[i])

            clf_features = np.concatenate(clf_features)
            if classifier == "LR":
                LR = linear_model.LogisticRegression()
                clf = LR.fit(clf_features, training_data.labels)
                joblib.dump(LR, "./models/" + model_type + "-" + str(epoch) + "-" + classifier + ".pkl")
            elif classifier == "SVM":
                SVM = svm.LinearSVC()
                clf = SVM.fit(clf_features, training_data.labels)
                joblib.dump(SVM, "./models/" + model_type + "-" + str(epoch) + "-" + classifier + ".pkl")

            print('=' * 25)
            print('testing......')

            testing_data.reset_index()
            QA_pairs = {}

            result = []
            while testing_data.is_available():
                batch_questions, batch_answers, batch_labels, batch_features = testing_data.next_batch(batch_size=batch_size)

                for i in range(len(batch_questions)):
                    pred, clf_input = sess.run([model.prediction, model.output_features],
                                               feed_dict={model.x1: np.expand_dims(batch_questions[i], axis=0),
                                                          model.x2: np.expand_dims(batch_answers[i], axis=0),
                                                          model.y: np.expand_dims(batch_labels[i], axis=0),
                                                          model.features: np.expand_dims(batch_features[i], axis=0)})

                    if np.isnan(clf_input).any():
                        print("nan clf_input")
                        clf_input = np.nan_to_num(clf_input)

                    if classifier == "LR":
                        clf_pred = clf.predict_proba(clf_input)[:, 1]
                        pred = clf_pred
                    elif classifier == "SVM":
                        clf_pred = clf.decision_function(clf_input)
                        pred = clf_pred

                    result.append(pred)

                    question = " ".join(testing_data.questions[i])
                    answer = " ".join(testing_data.answers[i])

                    if question in QA_pairs:
                        QA_pairs[question].append((answer, batch_labels[i], np.asscalar(pred)))
                    else:
                        QA_pairs[question] = [(answer, batch_labels[i], np.asscalar(pred))]

            with open("./results/result" + model_type + "-" + str(epoch) + "-" + classifier + ".txt", "w", encoding="utf-8") as f:
                for i in range(len(result)):
                    print(result[i][0], file=f)

            MAP, MRR = evaluation(QA_pairs)
            print("[Epoch " + str(epoch) + "] MAP:", MAP, "/ MRR:", MRR)
            MAPs.append(MAP)
            MRRs.append(MRR)

            print("testing finished!")
            print("=" * 25)

        print("training finished!")
        print("=" * 50)

    print("=" * 50)
    print("max MAP:", max(MAPs), "max MRR:", max(MRRs))
    print("=" * 50)

    with open("./results/" + classifier + ".txt", "w", encoding="utf-8") as f:
        print("Epoch\tMAP\tMRR", file=f)
        for i in range(epoch):
            print(str(i + 1) + "\t" + str(MAPs[i]) + "\t" + str(MRRs[i]), file=f)


# def display(file_path, classifier):
#     with open(file_path, "r", encoding="utf-8") as f:
#         f.readline()
#         MAPs, MRRs = [], []

#         for line in f:
#             MAP = line[:-1].split("\t")[1]
#             MRR = line[:-1].split("\t")[2]

#             MAPs.append(MAP)
#             MRRs.append(MRR)

#     print("max:", max(MAPs), max(MRRs))

#     plt.plot(np.arange(1, len(MAPs)+1, 1), MAPs, 'r')
#     plt.plot(np.arange(1, len(MAPs)+1, 1), MRRs, 'b')
#     plt.legend(["MAP", "MRR"])

#     plt.savefig("./results/" + classifier + '.png')
#     plt.show()

def test(testing_data, sentence_length, window_size, training_epochs, classifier, l2_reg, model_type, num_layers=2, num_classes=2):
    model = CNN_Attention(s=sentence_length,  w=window_size,  l2_reg=l2_reg, model_type=model_type, num_features=testing_data.num_features, num_classes=num_classes, num_layers=num_layers)

    
    for epoch in range(1, training_epochs + 1):

        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "./models/new" + model_type + "-" + str(epoch))

            testing_data.reset_index()
            result = []
            QA_pairs = {}
            while testing_data.is_available():
                batch_questions, batch_answers, batch_labels, batch_features = testing_data.next_batch(batch_size=batch_size)

                for i in range(len(batch_questions)):
                    pred, clf_input = sess.run([model.prediction, model.output_features],
                                               feed_dict={model.x1: np.expand_dims(batch_questions[i], axis=0),
                                                          model.x2: np.expand_dims(batch_answers[i], axis=0),
                                                          model.y: np.expand_dims(batch_labels[i], axis=0),
                                                          model.features: np.expand_dims(batch_features[i], axis=0)})

                    if np.isnan(clf_input).any():
                        print("nan clf_input")
                        clf_input = np.nan_to_num(clf_input)

                    if classifier == "LR":
                        clf_path = "./models/" + model_type + "-" + str(epoch) + "-" + classifier + ".pkl"
                        clf = joblib.load(clf_path)
                        clf_pred = clf.predict_proba(clf_input)[:, 1]
                        pred = clf_pred
                    elif classifier == "SVM":
                        clf_path = "./models/" + model_type + "-" + str(epoch) + "-" + classifier + ".pkl"
                        clf = joblib.load(clf_path)
                        clf_pred = clf.decision_function(clf_input)
                        pred = clf_pred

                    result.append(pred)

                    question = " ".join(testing_data.questions[i])
                    answer = " ".join(testing_data.answers[i])

                    if question in QA_pairs:
                        QA_pairs[question].append((answer, batch_labels[i], np.asscalar(pred)))
                    else:
                        QA_pairs[question] = [(answer, batch_labels[i], np.asscalar(pred))]

            with open("./results/result" + model_type + "-" + str(epoch) + "-" + classifier + ".txt", "w+", encoding="utf-8") as f:
                for i in range(len(result)):
                    print(result[i][0], file=f)


if __name__ == '__main__':
    learning_rate = 1e-5
    window_size = 4
    l2_reg = 0.0004
    training_epochs = 50
    batch_size = 16
    model_type = "ABCNN3"
    num_layers = 2
    sentence_length = 50
    classifier = "SVM"
    
    print("=" * 50)
    print("Parameters: learning_rate: ", learning_rate, ", window_size: ", window_size, ", l2_reg: ", l2_reg, ", training_epochs: ", 
        training_epochs, ", batch_size: ", batch_size, ", model_type: ", model_type, ", num_layers: ", num_layers, ", classifier: ", classifier)
    print("=" * 50)
    
    # nlpcc-iccpol-2016.db
    training_data = QA_pair('data/nlpcc-iccpol-2016.dbqa.training-data', sentence_length)
    testing_data = QA_pair('data/nlpcc-iccpol-2016.dbqa.testing-data', sentence_length)
    print("=" * 50)
    print("Training data size:", training_data.data_size, " Testing data size:", testing_data.data_size)
    print("Training max len:", training_data.max_len, " Testing max len:", testing_data.max_len)
    print("Training data features:", training_data.num_features, " Testing data features:", testing_data.num_features)
    print("=" * 50)

    train(training_data, testing_data, sentence_length, learning_rate=learning_rate, window_size=window_size, l2_reg=l2_reg, training_epochs=training_epochs, 
        batch_size=batch_size, classifier=classifier, model_type=model_type, num_layers=num_layers)

    # display("./results/" + classifier + ".txt", classifier)
   
    # test(testing_data, sentence_length, window_size, training_epochs, classifier, l2_reg, model_type, num_layers)
 

