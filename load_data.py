import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def load_path(train=True):
    data_path = []
    path = 'part of celebA'
    for filename in os.listdir(path):
        data_path.append(filename)
    train_path = data_path[0:10000]
    test_path = data_path[-100:]
    train_path = np.array(train_path)
    test_path = np.array(test_path)
    if train:
        return train_path
    else:
        return test_path


def get_batch_data(data,batch_idx,batch_size):
    range_min = batch_idx * batch_size
    range_max = (batch_idx + 1 ) * batch_size

    if range_max > len(data):
        range_max = len(data)
    index = list(range(range_min,range_max))
    train_data = [data[idx] for idx in index]
    return train_data


def load_image(roots,inpainting=False):
    train_data = []
    path = "part of celebA"
    for root in roots:
        img = cv2.imread(path + "/" + root,cv2.IMREAD_GRAYSCALE)
        if inpainting:
            img = cv2.rectangle(img,(27,27),(37,37),(0,0,0),-1)
        train_data.append(img)
    train_data = np.array(train_data)
    return (np.array(train_data).astype('float32')) / 127.5 - 1


if __name__ == "__main__":
    train_path = load_path(train=True)

