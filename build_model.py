from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf


def build_generator():
    input = Input((100))
    d1 = Dense(4*4*1024,use_bias=False,name="d1")(input)
    bat1 = BatchNormalization(name="bat1")(d1)
    ac1 = LeakyReLU(0.3,name="ac1")(bat1)
    reshape = Reshape((4,4,1024),name="reshape")(ac1)
    deconv1 = Conv2DTranspose(512,5,strides=2,padding="same",use_bias=False,name="deconv1")(reshape)
    bat2 = BatchNormalization(name="bat2")(deconv1)
    ac2 = LeakyReLU(0.3,name="ac2")(bat2)
    deconv2 = Conv2DTranspose(256,5,strides=2,padding="same",use_bias=False,name="deocnv2")(ac2)
    bat3 = BatchNormalization(name="bat3")(deconv2)
    ac3 = LeakyReLU(0.3,name="ac3")(bat3)
    deconv3 = Conv2DTranspose(128,5,strides=2,padding="same",use_bias=False,name="deconv3")(ac3)
    bat4 = BatchNormalization(name="bat4")(deconv3)
    ac4 = LeakyReLU(0.3,name="ac4")(bat4)
    output = Conv2DTranspose(1,5,strides=2,padding="same",use_bias=False,activation="tanh",name="output")(ac4)
    model = Model(input,output)
    model.summary()
    return model


def build_discriminator():
    input = Input((64,64,1))
    conv1 = Conv2D(64,5,activation=LeakyReLU(0.3),strides=2,padding="same",name="conv1")(input)
    drop1 = Dropout(0.3,name="drop1")(conv1)
    conv2 = Conv2D(128,5,activation=LeakyReLU(0.3),strides=2,padding="same",name="conv2")(drop1)
    drop2 = Dropout(0.3,name="drop2")(conv2)
    flat = Flatten(name="flat")(drop2)
    output = Dense(1,activation="softmax",name="output")(flat)
    model = Model(input,output)
    model.summary()
    return model


if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()

