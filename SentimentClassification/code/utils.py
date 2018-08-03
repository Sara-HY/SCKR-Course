import regex as re
import numpy as np
from nltk import word_tokenize
import tensorflow.contrib.keras as kr
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = " {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["_hashtag_"] + re.split(r"(?=[A-Z])", hashtag_body, flags=re.MULTILINE | re.DOTALL))
    return result


def allcaps(text):
    text = text.group()
    return text.lower() + " _allcaps_"


def preprocess_one_tweet(text):
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=re.MULTILINE | re.DOTALL)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "_url_")
    text = re_sub(r"@\w+", "_user_")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "_smile_")
    text = re_sub(r"{}{}p+".format(eyes, nose), "_lolface_")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "_sadface_")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "_neutralface_")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","_heart_")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "_number_")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 _repeat_")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 _elong_")

    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()


def preprocess(texts):
    return [preprocess_one_tweet(x) for x in texts]


def read_data(file_path, verobse=True):
    label_encoding = dict({'negative': 0, 'neutral': 1, 'positive': 2})
    pfile = open(file_path, "r")
    data = pfile.read().split('\n')
    pfile.close()

    if len(data[-1]) == 0:
        del data[-1]

    data = list(map(lambda x: x.split('\t'), data))

    # deal with the tweets lost quotation
    lost_quotation_marks = [i for i,x in enumerate(data) if x[0] == "\"" ]
    for i in lost_quotation_marks:
        data[i-1].append("\"")
    data = [x for x in data if x[0] != "\""]

    # map the data to (category, text)
    data = map(lambda x: (x[1], " ".join((" ".join(x[2:])).split())), data)
    data = list(map(lambda x: (x[0], x[1].replace('""','"')), list(data)))

    # split the data to x, y
    X = [x[1][1:-1] if (x[1][0]=='"' and x[1][-1]=='"') else x[1] for x in data]

    X = preprocess(X)
    y = [label_encoding[x[0]] for x in data]

    if verobse > 0:
        print("Data report:\n", "Number of obserwations: ", len(X))
    return X, y


def load_embeddings(embedding_path, emb_dim):
    words2ids = {}
    vectors = [np.zeros(emb_dim)]
    i = 1
    with open(embedding_path, "r") as f:
        for line in f:
            tokens = line.split(" ")
            word = tokens[0]
            if True:
                v = list(map(float, tokens[1:]))
                vectors.append(v)
                words2ids[word] = i
                i = i + 1

    vectors = np.array(vectors)
    words2ids["_unknown_"] = 0

    # np.random.uniform(-0.25, 0.25, emb_dim)

    # keys = list(words2ids.keys())
    #
    # for key in keys:
    #     if key[0]=="<" and key[-1]==">":
    #
    #         new_key = "_" + key[1:-1] + "_"
    #         words2ids[new_key] = words2ids.pop(key)
    return words2ids, vectors


def transform(X, y, words2ids, maxlen):
    X_new = [[words2ids.get(x, words2ids["_unknown_"]) for x in word_tokenize(X[i])] for i in range(len(X))]

    X_new = kr.preprocessing.sequence.pad_sequences(np.array(X_new), value=0., maxlen=maxlen)

    y_new = to_categorical(y)

    return X_new, y_new

# display the learning curve
def dispaly_results(title, train_cost, train_accuracy, valid_cost, valid_accuracy, step):
    training_iters = len(train_cost)
    # iters_steps
    iter_steps = [step *k for k in range(training_iters)]
    
    imh = plt.figure(1, figsize=(8, 6), dpi=160)

    imh.suptitle(title)
    plt.subplot(221)
    plt.semilogy(iter_steps, train_cost, '-g', label='Train Loss')
    plt.title('Train Loss ')
    plt.legend(loc='upper right')
    
    plt.subplot(222)
    plt.plot(iter_steps, train_accuracy, '-r', label='Train Accuracy')
    plt.title('Train Accuracy')
    plt.legend(loc='upper right')

    plt.subplot(223)
    plt.semilogy(iter_steps, valid_cost, '-g', label='Test Loss')
    plt.title('Valid Loss')
    plt.legend(loc='upper right')
    
    plt.subplot(224)
    plt.plot(iter_steps, valid_accuracy, '-r', label='Test Accuracy')
    plt.title('Valid Accuracy')
    plt.legend(loc='upper right')


    #plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plot_file = ".result/{}.png".format(title.replace(" ","_"))
    plt.savefig(plot_file)
    plt.show()
    