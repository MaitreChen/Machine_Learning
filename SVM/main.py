from time import perf_counter
import numpy as np
import pickle
import os

from sklearn.svm import SVC

from utils import load_mnist, data2libsvm, feature_extraction, getConfig


# 保存libsvm格式的文件，用于libsvm工具包训练
def save_txt(X, Y, filename):
    data_set = data2libsvm(X, Y)
    with open(filename, 'w') as f:
        for data in data_set:
            f.write(data + '\n')


# 提取hog特征
def get_hog_feature(X, T):
    out_print = 'training set' if T == 'train' else 'testing set'
    print(f"\nStart extracting the {out_print} features......")

    t = perf_counter()
    feature = []
    for x in X:
        ft = feature_extraction(x.reshape(28, 28), stride=1)
        feature.append(ft)

    print("Feature extraction completed......")
    print(f"Time consumption: {perf_counter() - t:.2f}s")

    if T == 'train':
        np.save(train_feature_path, np.array(feature))
    else:
        np.save(test_feature_path, np.array(feature))

    return np.array(feature, 'float64')


def get_hog_feature_official(X):
    from skimage.feature import hog
    feature = []
    for x in X:
        ft = hog(x.reshape((28, 28)), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2))
        feature.append(ft)
    return np.array(feature, 'float64')


def train(X, y):
    t = perf_counter()

    model = SVC(C=C, gamma=gamma,kernel=kernel)
    model.fit(X, y.ravel())

    print(f'Training time: {perf_counter() - t:.2f}s')

    # saving
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def test(X, y):
    t = perf_counter()

    # loading
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        prediction = model.predict(X)
        print(f'Testing time: {perf_counter() - t:.2f}s')

        nums_of_correct = sum(np.array(prediction.reshape(-1, 1)) == y)[0]
        print(f"Accuracy: {100. * nums_of_correct / len(X):.2f}%")

        # 为了进一步观察错误样本的个数以及每一类错分的样本，可以结合混淆矩阵查看
        """
        from sklearn.metrics import confusion_matrix
        print(f"Nums of wrong samples: {len(X) - nums_of_correct}")

        cf_matrix = confusion_matrix(prediction.reshape(-1, 1), y, labels=[i for i in range(10)])
        print(f"\nconfusion_matrix:\n {cf_matrix}")

        each_samples = sum(cf_matrix[i, :] for i in range(10))
        each_error = [(each_samples[i] - cf_matrix[i][i]) for i in range(10)]
        print(f'Number of samples per class: {each_samples}')
        print(f'Number of error samples per class: {each_error}')
        """


if __name__ == '__main__':
    # GET CONSTANT CONFIGURATION
    cfg_path = 'config.cfg'

    C = eval(getConfig(cfg_path, 'hyper-parameter', 'C'))
    gamma = eval(getConfig(cfg_path, 'hyper-parameter', 'gamma'))
    kernel = getConfig(cfg_path, 'hyper-parameter', 'kernel')

    train_data_path = getConfig(cfg_path, 'path', 'train_data_path')
    train_label_path = getConfig(cfg_path, 'path', 'train_label_path')
    test_data_path = getConfig(cfg_path, 'path', 'test_data_path')
    test_label_path = getConfig(cfg_path, 'path', 'test_label_path')

    feature_dir = getConfig(cfg_path, 'feature', 'feature_dir')
    train_feature_name = getConfig(cfg_path, 'feature', 'train_feature_name')
    test_feature_name = getConfig(cfg_path, 'feature', 'test_feature_name')
    train_feature_path = os.path.join(feature_dir, train_feature_name).replace('\\', '/')
    test_feature_path = os.path.join(feature_dir, test_feature_name).replace('\\', '/')

    model_dir = getConfig(cfg_path, 'model', 'model_dir')
    model_name = getConfig(cfg_path, 'model', 'model_name')
    model_path = os.path.join(model_dir, model_name).replace('\\', '/')

    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # 1.Load dataset and preprocessing
    train_data, train_label = load_mnist(train_label_path, train_data_path)
    test_data, test_label = load_mnist(test_label_path, test_data_path)

    train_label = train_label.reshape(-1, 1)
    test_label = test_label.reshape(-1, 1)

    print(f'Training shape: {train_data.shape, train_label.shape}')
    print(f'Testing shape: {test_data.shape, test_label.shape}')

    # 2.Extract feature
    if os.path.exists(train_feature_path):
        train_feature = np.load(train_feature_path)
    else:
        train_feature = get_hog_feature(train_data, 'train')

    if os.path.exists(test_feature_path):
        test_feature = np.load(test_feature_path)
    else:
        test_feature = get_hog_feature(test_data, 'test')

    print(f"\nFeature shape of the training data: {train_feature.shape}")
    print(f"Feature shape of the testing data: {test_feature.shape}")

    # 3.Train and Save model
    print(f"\nStart training......")
    train(train_feature, train_label)

    # 4.Test
    print(f"\nStart testing ......")
    test(test_feature, test_label)
