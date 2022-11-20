"""
123 features of each sample
and the shape of sample is (1,123)
Here are 1605 training sample and 30956 testing sample
"""

from sklearn.datasets import load_svmlight_files
import numpy as np
import os
from matplotlib import pyplot as plt
from setting import *


def process_samples(X, Y, T):
    """
    :param X: 样本集(train or test)，密集矩阵
    :param Y: 样本对应的标签label
    :param T: 存储方式 (密集矩阵存储 or 稀疏存储)
    :return: 以libsvm格式构成的列表
    """
    data_set = []
    for x_sample, y_sample in zip(X, Y):
        data = ''
        data += str(-1) if int(y_sample) == 0 else '+' + str(+1)

        if T == 'sparse':  # 以稀疏阵存储
            for i, v in enumerate(x_sample):
                if v == 0:
                    continue
                data += ' ' + str(i + 1) + ':' + str(v)
            data_set.append(data)
        elif T == 'dense':
            for i, v in enumerate(x_sample):
                data += ' ' + str(i + 1) + ':' + str(v)
            data_set.append(data)

    return data_set


# 存储样本数据集
def save_txt(X, Y, T, filename):
    data_set = process_samples(X, Y, T)
    with open(filename, 'w') as f:
        for data in data_set:
            f.write(data + '\n')


def sigmoid(x):
    """
    x非常小的时，即作为负数，exp(-x)过大存在溢出，因此，
    原式分子、分母同乘一个较小的数字，这里采用exp(x)
    """
    if x.all() >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


# 回归方程
def logistic_regression(X, w, b):
    return sigmoid(np.dot(X, w) + b)


def load_dataset(filename, nums_of_samples):
    sample_data = np.zeros(
        (nums_of_samples, num_features), dtype=int)
    sample_label = np.zeros(nums_of_samples)

    with open('dataset/' + filename, 'r') as f:
        for id, line in enumerate(f.readlines()):
            line_split = line.replace(':1', '').split()
            feature, label = line_split[1:], line_split[0]
            sample_label[id] = label if label == '+1' else 0  # attention
            feature = list(map(int, feature))
            for i in range(1, num_features + 1):
                sample_data[id][i - 1] = 1 if i in feature else 0

    sample_label = sample_label.T.reshape(sample_label.shape[0],
                                          1)  # dimension conversion,such as a1a(1605,)--->(1605,1)

    return sample_data, sample_label


def init_params(nums_of_feature):
    w = np.ones((nums_of_feature, 1))
    b = 0
    return w, b


def train(X, y):
    w, b = init_params(num_features)

    train_loss_set = []
    for epoch in range(epochs):
        y_hat = logistic_regression(X, w, b)
        loss = -1 / nums_of_training * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        w = w - alpha * (np.dot(X.transpose(), (y_hat - y)))
        b = b - alpha * np.sum(y_hat - y)

        train_loss_set.append(loss)

        if epoch % log_interval == 0:
            print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss))

        state_dict = {'weight': w, 'bias': b}

    return state_dict, train_loss_set


def test(X, y, params):
    w, b = params['weight'], params['bias']

    predict = logistic_regression(X, w, b)
    for i, v in enumerate(predict):
        predict[i] = 1 if v > 0.5 else 0

    correct = np.sum(predict == y)
    return 100. * correct / len(predict)


def plot_training_loss(loss_set):
    """
    为了观察方便，这里选取前20个epoch的结果进行再一次可视化
    """
    plt.figure(figsize=(10, 6))
    plt.rc('font', family='Times New Roman')

    plt.subplot(121)
    plt.title("Training loss", fontsize=20)
    idx = [_ for _ in range(len(loss_set))]
    plt.plot(idx, loss_set, label=data_name)
    plt.xlabel('Epochs', fontsize=20), plt.ylabel('Loss', fontsize=20)
    plt.xticks(fontsize=20), plt.yticks(fontsize=20)
    plt.legend(fontsize=20)

    plt.subplot(122)
    plt.title("Training loss", fontsize=20)
    plt.plot(idx[:20], loss_set[:20], label=data_name)
    plt.xlabel('Epochs', fontsize=20), plt.ylabel('Loss', fontsize=20)
    plt.xticks(fontsize=20), plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('res/loss/' + data_name + '_loss.png', dpi=300)
    plt.show()


def call_sklearn(name):
    from sklearn.linear_model import LogisticRegression
    X_train, Y_train, X_test, Y_test = load_svmlight_files(
        ("../data/UCI/" + name + ".txt", "../data/UCI/" + name + ".t"))

    modelLR = LogisticRegression(max_iter=300)  # attention
    modelLR.fit(X_train, Y_train)
    accuracy = modelLR.score(X_test, Y_test)

    print(f"\nSklearn   Test accuracy: {100. * accuracy:.2f}%")


def create_space(name):
    features = 123
    train_num, test_num = data[name][0], data[name][1]
    return train_num, test_num, features


if __name__ == '__main__':
    log_loss_path = 'res/loss/'
    if not os.path.isdir(log_loss_path):
        os.makedirs(log_loss_path)

    data_name = 'a9a'
    log_interval = 20
    epochs = hyper_params[data_name]['epoch']
    alpha = hyper_params[data_name]['alpha']

    # 1.获取样本属性
    nums_of_training, nums_of_testing, num_features = create_space(data_name)

    # 2.加载数据集
    train_data, train_label = load_dataset(data_name + ".txt", nums_of_training)
    test_data, test_label = load_dataset(data_name + ".t", nums_of_testing)

    # 以dense或sparse格式重新存储数据集
    # if not os.path.isdir('res/data/'):
    #     os.makedirs('res/data/')
    # save_txt(train_data, train_label, 'dense', 'res/data/training_dense.txt')
    # save_txt(test_data, test_label, 'sparse', 'res/data/testing_sparse.txt')

    # 3.训练
    print("Start training on " + data_name)
    model, train_loss = train(train_data, train_label)
    plot_training_loss(train_loss)

    # 4.测试
    # print(f"\nTrain set: accuracy: {test(train_data, train_label, model):.2f}%")
    print(f"Test set: accuracy: {test(test_data, test_label, model):.2f}%")

    # 5.调用sklearn库，性能对比
    call_sklearn(data_name)
