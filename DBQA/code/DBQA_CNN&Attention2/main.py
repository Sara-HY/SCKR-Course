from config import opt
import torch as t
import models
from data import Small
from torch.autograd import Variable
from torch.optim import Adamax
from utils.visualize import Visualizer
from tqdm import tqdm
from utils.eval import MAP, MRR
import os


def test(**kwargs):
    opt.parse(kwargs)
    # data
    test_data = Small(opt.test_root,
                      wv_path=opt.word2vec_path,
                      stopwords_path=opt.stopwords_path,
                      idf_path=opt.idf_test_path,
                      test=True)
    # configure model
    model = getattr(models, opt.model)(opt).eval()
    if opt.load_model_path:
        if os.path.isdir(opt.load_model_path):
            for file_name in os.listdir(opt.load_model_path):
                model.load(os.path.join(opt.load_model_path, file_name))
                if opt.use_gpu:
                    model.cuda()
                score_list = test_(model, test_data)
                write_result(score_list, os.path.join(opt.result_dir, file_name + '.out'))
            return
        else:
            model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()
    score_list = test_(model, test_data)
    write_result(score_list, os.path.join(opt.result_dir, model.module_name + '.out'))


def test_(model, test_data):
    score_list = []
    for ii, (q, a, label, s) in tqdm(enumerate(test_data)):
        val_input_q = Variable(q, requires_grad=False)
        val_input_a = Variable(a, requires_grad=False)
        s = Variable(s, requires_grad=False)
        if opt.use_gpu:
            val_input_q = val_input_q.cuda()
            val_input_a = val_input_a.cuda()
            s = s.cuda()
        score = model(val_input_q, val_input_a, s)
        score_list.extend(score.data.tolist())

    return score_list


def write_result(result, file_path):
    with open(file_path, 'w') as f:
        f.write('\n'.join(str.format('%.6f', x) for x in result))


def train(**kwargs):
    opt.parse(kwargs)
    if opt.vis:
        vis = Visualizer(opt.env)

    # step 1: configure model
    model = getattr(models, opt.model)(opt)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    # step 2: data
    train_data = Small(opt.train_root,
                       wv_path=opt.word2vec_path,
                       stopwords_path=opt.stopwords_path,
                       idf_path=opt.idf_train_path,
                       train=True)
    # val_data = Small(opt.train_root,
    #                  wv_path=opt.word2vec_path,
    #                  stopwords_path=opt.stopwords_path,
    #                  train=False)

    data_size = len(train_data)
    indices = t.randperm(data_size)

    # step 3: criterion and optimizer
    criterion = t.nn.KLDivLoss()
    lr = opt.lr
    optimizer = Adamax(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    # step 4: meters
    previous_loss = float('inf')

    # train
    for epoch in range(opt.max_epoch):

        for i in tqdm(range(0, data_size, opt.batch_size)):

            batch_size = min(opt.batch_size, data_size - i)
            # train_model
            loss = 0.
            for j in range(0, batch_size):
                idx = indices[i+j]
                q, a, label, shallow_features = train_data[idx]
                input_q, input_a, shallow_features = Variable(q), Variable(a), Variable(shallow_features)
                target = Variable(label)
                if opt.use_gpu:
                    input_q = input_q.cuda()
                    input_a = input_a.cuda()
                    shallow_features = shallow_features.cuda()
                    target = target.cuda()

                score = model(input_q, input_a, shallow_features)
                example_loss = criterion(score, target)
                loss += example_loss
            loss /= opt.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.save(model.module_name + '_' + str(epoch) + '.pth')

        print('epoch:{epoch}, lr:{lr}, loss:{loss}'.format(
            epoch=epoch,
            loss=loss.data,
            lr=lr
        ))

        # # validate and visualize
        # map, mrr = val(model, val_data)
        #
        # print('epoch:{epoch}, lr:{lr}, loss:{loss}, map:{map}, mrr:{mrr}'.format(
        #     epoch=epoch,
        #     loss=loss.data,
        #     map=map,
        #     mrr=mrr,
        #     lr=lr
        # ))

        # update learning rate
        if (loss.data > previous_loss).all():
            lr = lr * opt.lr_decay

        previous_loss = loss.data


def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''
        usage : python file.py <function> [--args=value]
        <function> := train | test | help
        example: 
                python {0} train --env='env0701' --lr=0.01
                python {0} test --dataset='path/to/dataset/root/'
                python {0} help
        avaiable args:'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print('\t\t{0}: {1}'.format(k, v))


if __name__ == '__main__':
    import fire
    fire.Fire()
