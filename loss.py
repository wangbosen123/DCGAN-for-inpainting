import tensorflow as tf
from tensorflow.keras.losses import *
import numpy as np
from load_data import *
from build_model import *



def create_mask():
    train_path = load_path()
    train_data = load_image(get_batch_data(train_path,0,1),inpainting=True)
    train_data = train_data.reshape(64,64)
    mask = []
    for row, i in enumerate(train_data):
        for col ,j in enumerate(i):
            idx = 0
            if train_data[row][col] == -1:
                mask.append(0)
            elif row==0 or row==63 or col == 0 or col ==63:
                mask.append(0)
            else:
                if train_data[row][col-1] == -1:
                    idx+=1
                if train_data[row][col+1] == -1:
                    idx += 1
                if train_data[row-1][col-1] == -1:
                    idx += 1
                if train_data[row-1][col] == -1:
                    idx += 1
                if train_data[row-1][col+1] == -1:
                    idx += 1
                if train_data[row+1][col-1] == -1:
                    idx += 1
                if train_data[row+1][col] == -1:
                    idx += 1
                if train_data[row+1][col+1] == -1:
                    idx += 1
                if idx ==0:
                    mask.append(0)
                else:
                    mask.append(idx)

    mask = (1/9) * np.array(mask)
    mask = mask.reshape(64,64)
    return mask


def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def image_loss(real,pred):
    mae = tf.keras.losses.MeanAbsoluteError()
    return mae(real,pred)


def context_loss(real,fake):
    mask = create_mask()
    mae = tf.keras.losses.MeanAbsoluteError()
    minus = tf.zeros((64,64))
    return mae(mask*(fake-real),minus)

def prior_loss(pred,lambdA=0.03):
    return lambdA * tf.math.log(1-pred)





