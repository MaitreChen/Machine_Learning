"""
该工具库包含了以下功能：
mnist数据集的读取、将数据保存为libsvm格式、hog特征提取
"""

import numpy as np
import cv2 as cv
import struct


def load_mnist(labels_path, images_path):
    with open(labels_path, 'rb') as label_path:
        magic, n = struct.unpack('>ii', label_path.read(8))
        labels = np.fromfile(label_path, dtype=np.uint8)
    with open(images_path, 'rb') as image_path:
        magic, num, rows, cols = struct.unpack('>iiii', image_path.read(16))
        images = np.fromfile(image_path, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def parse_idx3_ubyte_file(path):
    # 1 read ubyte file
    bin_data = open(path, 'rb').read()

    # 2.Parse the head of file from offset 0
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, rows, cols = struct.unpack_from(fmt_header, bin_data, offset)

    # 3.parse dataset
    img_size = rows * cols
    offset += struct.calcsize(fmt_header)  # 计算fmt格式占用的内存数(byte)
    fmt_image = '>' + str(img_size) + 'B'

    data_set = np.empty((num_images, rows * cols))  # open the empty to restore images
    for i in range(num_images):
        data_set[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        offset += struct.calcsize(fmt_image)
    return data_set


def parse_idx1_ubyte_file(path):
    # 1 read ubyte file
    bin_data = open(path, 'rb').read()

    # 2.Parse the head of file from offset 0
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)

    # 3.parse labels
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + 'B'
    label_set = np.empty((num_images, 1))
    for i in range(num_images):
        label_set[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset))
        offset += struct.calcsize(fmt_image)
    return label_set


def data2libsvm(X, Y):
    data_set = []
    for x_sample, y_sample in zip(X, Y):
        data = ''
        data += str(y_sample)

        for i, v in enumerate(x_sample):
            if v == 0:
                continue
            data += ' ' + str(i + 1) + ':' + "%.7f" % (v / 255)
        data_set.append(data)

    return data_set


def feature_extraction(src, cell_size=7, bin_size=9, stride=1):
    # Params setting
    h, w = src.shape
    src = src / 255.0
    # 1.Compute the gradient magnitude and angle
    # grad_x = np.zeros((w, h), dtype=np.float)
    # grad_y = np.zeros((w, h), dtype=np.float)
    # magnitude = np.zeros((w, h), dtype=np.float)
    # angle = np.zeros((w, h), dtype=np.float)

    # for i in range(1, h - 1):
    #     for j in range(1, w - 1):
    #         gx = src[i][j + 1] - src[i][j - 1]
    #         gy = src[i + 1][j] - src[i - 1][j]
    #         grad_x[i, j] = gx
    #         grad_y[i, j] = gy
    #
    #         magnitude[i, j] = np.sqrt(gx ** 2 + gy ** 2)
    #         angle[i, j] = np.arctan(gy / (gx + 1e-5)) * 180 / pi
    grad_x = cv.Sobel(src, cv.CV_64F, dx=1, dy=0, ksize=3)
    grad_y = cv.Sobel(src, cv.CV_64F, dx=0, dy=1, ksize=3)
    magnitude, angle = cv.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    angle[angle > 180] -= 180

    # 2.Compute the histogram of each cell
    cell_accumulator = np.zeros((int(h / cell_size), int(w / cell_size), bin_size), dtype=float)  # 4*4*9
    for i in range(cell_accumulator.shape[0]):
        for j in range(cell_accumulator.shape[1]):
            cell_magnitude = magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
            cell_angle = angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]

            # prepared to send to bin
            bin_accumulator = np.zeros(bin_size)
            for p in range(cell_size):
                for q in range(cell_size):
                    grand_magnitude = cell_magnitude[p][q]
                    idx, n = divmod(cell_angle[p][q], 20)
                    idx = int(idx)
                    bin_accumulator[idx % bin_size] += (1 - n / 20) * grand_magnitude
                    bin_accumulator[(idx + 1) % bin_size] += (n / 20) * grand_magnitude
            # print(bin_accumulator)
            cell_accumulator[i][j] = bin_accumulator

    # 3.combine cell to get block(2*2)
    hog_feature = []
    for i in range(0, cell_accumulator.shape[0] - 1, stride):
        for j in range(0, cell_accumulator.shape[1] - 1, stride):
            block = cell_accumulator[j:j + 2, i:i + 2].ravel()

            # normalization based L2-norm
            sum_block = sum(i * i for i in block) ** 0.5
            block_norm = block / sum_block if sum_block != 0 else block
            hog_feature.extend(block_norm)

    return np.array(hog_feature).ravel()
