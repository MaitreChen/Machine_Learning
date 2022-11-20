from time import perf_counter
import pickle
import os
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

from utils import *


def save_txt(X, Y, filename):
    data_set = data2libsvm(X, Y)
    with open(filename, 'w') as f:
        for data in data_set:
            f.write(data + '\n')


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

    np.save(feature_path + T, np.array(feature))
    return np.array(feature, 'float64')


# def get_hog_feature_official(X):
#     from skimage.feature import hog
#     feature = []
#     for x in X:
#         ft = hog(x.reshape((28, 28)), orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2))
#         feature.append(ft)
#     return np.array(feature, 'float64')


def train(X, y):
    t = perf_counter()

    model = SVC(C=6, gamma=0.25)
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
        """
        print(f"Nums of wrong samples: {len(X) - nums_of_correct}")

        cf_matrix = confusion_matrix(prediction.reshape(-1, 1), y, labels=[i for i in range(10)])
        print(f"\nconfusion_matrix:\n {cf_matrix}")

        each_samples = sum(cf_matrix[i, :] for i in range(10))
        each_error = [(each_samples[i] - cf_matrix[i][i]) for i in range(10)]
        print(f'Number of samples per class: {each_samples}')
        print(f'Number of error samples per class: {each_error}')
        """


if __name__ == '__main__':
    # CONSTANT
    train_path = '../data/MNIST/train-images-idx3-ubyte'
    train_label_path = '../data/MNIST/train-labels-idx1-ubyte'
    test_path = '../data/MNIST/t10k-images-idx3-ubyte'
    test_label_path = '../data/MNIST/t10k-labels-idx1-ubyte'

    if not os.path.isdir('model'):
        os.makedirs('model')

    if not os.path.isdir('feature'):
        os.makedirs('feature')

    model_path = 'model/best.pickle'
    feature_path = 'feature/ft_'

    # 1.Load dataset
    train_data, train_label = load_mnist(train_label_path, train_path)
    test_data, test_label = load_mnist(test_label_path, test_path)

    train_label = train_label.reshape(-1, 1)
    test_label = test_label.reshape(-1, 1)

    print(f'Training shape: {train_data.shape, train_label.shape}')
    print(f'Testing shape: {test_data.shape, test_label.shape}')

    # 2.Extract feature
    train_feature = get_hog_feature(train_data, 'train')
    test_feature = get_hog_feature(test_data, 'test')
    # train_feature = np.load(feature_path + 'train.npy')
    # test_feature = np.load(feature_path + 'test.npy')

    print(f"\nFeature shape of the training data: {train_feature.shape}")
    print(f"Feature shape of the testing data: {test_feature.shape}")

    # 3.Train and Save model
    print(f"\nStart training......")
    train(train_feature, train_label)

    # 4.Test
    print(f"\nStart testing ......")
    test(test_feature, test_label)
