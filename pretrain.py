import numpy as np
import tensorflow as tf
from build_model import *
from tensorflow.keras.optimizers import *
import time
from loss import *
from load_data import *

def train_step(data,batch_size,g_times,d_times):
    noise = tf.random.normal([batch_size,100])
    for i in range(g_times):
        with tf.GradientTape() as gen_tape:
            gen_img = generator(noise)
            fake = discriminator(gen_img)
            adv_loss = generator_loss(fake)
            img_loss = image_loss(data,gen_img)
            gen_loss = adv_loss + img_loss
        gradient_generaor = gen_tape.gradient(gen_loss,generator.trainable_variables)
        generator_optimizer.apply.gradients(zip(gradient_generaor,generator.trainable_variables))

    for i in range(d_times):
        with tf.GradientTape() as dis_tape:
            gen_img = generator(noise)
            real = discriminator(data)
            fake = discriminator(gen_img)
            dis_loss = discriminator_loss(real,fake)
        gradient_discriminator = dis_tape.gradient(dis_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply.gradients(zip(gradient_discriminator, discriminator.trainable_variables))

    return gen_loss , dis_loss , img_loss

def train(epochs,batch_size,batch_num,g_times,d_times):
    g_loss = []
    d_loss = []
    img_loss = []
    g_loss_avg = []
    d_loss_avg = []
    img_loss_avg = []
    for i in range(epochs):
        start = time.time()
        if i >250:
            generator_optimizer = tf.keras.optimizers.Adam(1e-4)
            discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        for j in range(batch_num):
            train_data = load_image(get_batch_data(train_path,j,batch_size))
            gen_loss , dis_loss , pic_loss = train_step(train_data,batch_size,g_times,d_times)
            gen_loss = np.array(gen_loss)
            dis_loss = np.array(dis_loss)
            g_loss.append(gen_loss)
            d_loss.append(dis_loss)
            img_loss.append(pic_loss)



        g_loss_avg.append(np.mean(g_loss))
        d_loss_avg.append(np.mean(d_loss))
        img_loss_avg.append(np.mean(img_loss))
        end = time.time()
        print("------------------------")
        print(f"the epochs:{i+1} ")
        print(f"the generator_loss is : {g_loss_avg[-1]}")
        print(f"the discriminator_loss is : {d_loss_avg[-1]}")
        print(f"the image_loss is : {img_loss_avg[-1]}")
        print("the spend time is : %s seconds " % (end - start))
        if i >270:
            generator.save_weights(f"model_weight/generator_{i+1}_weights")
            discriminator.save_weights(f"model_weight/discriminator_{i+1}_weights")
        draw_the_samples(i+1, generator)

    return g_loss_avg, d_loss_avg , img_loss_avg





def draw_the_samples(epochs,model,path="result_image/"):
    noise = tf.random.normal([10,100])
    image = model(noise)
    image = tf.reshape(image,[-1,64,64])
    data_path = load_path()
    data = load_image(get_batch_data(data_path,epochs,10))
    data = data.reshape(-1,64,64)
    ax,fig = plt.subplots(figsize=(15,4))

    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(image[i],cmap="gray")
        plt.subplot(2,10,i+11)
        plt.imshow(data[i],cmap="gray")
    plt.savefig(path + f"produce_image_{epochs}epochs")
    plt.close()







if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)
    train_path  = load_path(train=True)
    g_loss, d_loss , img_loss = train(500, 100, 100 ,1, 2)

    plt.plot(g_loss)
    plt.plot(d_loss)
    plt.legend()
    plt.title("the adversail loss for the generator and discriminator")
    plt.savefig("result_image/adversail_loss.jpg")
    plt.close()

    plt.plot(img_loss)
    plt.title("the image loss ")
    plt.savefig("result_image/image_loss.jpg")
    plt.close()
