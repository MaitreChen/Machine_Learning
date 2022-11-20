import numpy as np
import struct


def load_mnist(labels_path, images_path):
    with open(labels_path, 'rb') as label_path:
        magic, n = struct.unpack('>ii', label_path.read(8))  # 数据结构中前2行的数据类型都是32位整型，所以采用i格式
        labels = np.fromfile(label_path, dtype=np.uint8)
    with open(images_path, 'rb') as image_path:
        magic, num, rows, cols = struct.unpack('>iiii', image_path.read(16))
        images = np.fromfile(image_path, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


# 欧式距离
def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


# 曼哈顿距离
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# 余弦相似度
def cosine_similarity(p1, p2):
    d1 = np.sqrt(sum([i ** 2 for i in p1]))
    d2 = np.sqrt(sum([i ** 2 for i in p2]))
    return np.dot(p1, p2) / (d1 + d2)
