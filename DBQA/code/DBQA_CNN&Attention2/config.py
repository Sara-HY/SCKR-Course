import warnings


class DefaultConfig(object):
    vis = False
    env = 'main'  # visdom environment
    model = 'CompAggrModel'  # the name of the model to use

    train_root = './data/small/nlpcc-iccpol-2016.dbqa.training-data'
    test_root = './data/test.txt'
    load_model_path = None

    batch_size = 32
    use_gpu = True
    num_workers = 4  # how many workers for loading data
    print_freq = 2  # print info every N epoch

    max_epoch = 20
    lr = 0.001  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr *= lr_decay
    weight_decay = 0.05  # loss function

    word2vec_path = './resources/hanlp-wiki-vec-zh.txt'
    stopwords_path = './resources/stopwords.txt'
    idf_train_path = './resources/idf.train.pkl'
    idf_test_path = './resources/idf.test.pkl'
    result_dir='result'

    max_q_length = 50  # max query length
    max_a_length = 50  # max answer length
    emb_dim = 300
    kernel_sizes = [1, 2, 3, 4, 5]

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
