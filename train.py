import tensorflow as tf
from build_model import *
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import *
from load_data import *
from loss import *
import time



def train_step(code,target):
    code = tf.Variable(code)
    # gradient_code = []
    with tf.GradientTape(persistent=True) as code_tape:
        code_tape.watch(code)
        gen_img = generator(code)
        fake = discriminator(gen_img)
        gen_img = tf.reshape(gen_img, [64, 64])
        con_loss = context_loss(gen_img ,target)
        pr_loss = generator_loss(fake)
        total_loss = 10*con_loss + pr_loss

    gradient_code = code_tape.gradient(total_loss,code)
    code_optimizer.apply_gradients(zip([gradient_code],[code]))

    return 100*con_loss , pr_loss , code




def train(epochs,code,target):
    start = time.time()
    context_loss = []
    prior_loss = []
    print(f"the start time is {start}")
    for epoch in range(epochs):
        start = time.time()
        con_loss , pr_loss , code = train_step(code,target)
        context_loss.append(con_loss)
        prior_loss.append(pr_loss)
        end = time.time()
        print("______________________________________")
        print(f"the epoch is {epoch+1}")
        print(f"the context_loss is {context_loss[-1]}")
        print(f"the prior_loss is {prior_loss[-1]}")
        print(f"the new_code is {code[0][0:10]}")
        print("the spend time is %s second" %(end-start))
        draw_sample(epoch+1,code,target)

    return context_loss , prior_loss, code

def draw_sample(epoch,code,target, path = "result_image_inpainting3"):
    gen_image = generator(code)
    gen_image = tf.reshape(gen_image,[64,64])
    ax,fig = plt.subplots(figsize=(10,4))
    plt.subplot(2,1,1)
    plt.imshow(gen_image,cmap="gray")
    plt.subplot(2,1,2)
    plt.imshow(target,cmap="gray")
    plt.savefig(path + "/" + f"{epoch}.jpg")
    plt.close()






if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()
    code_optimizer = tf.keras.optimizers.Adam(1e-2)
    generator.load_weights("model_weight/generator_899_weights")
    discriminator.load_weights("model_weight/discriminator_899_weights")
    code = tf.random.uniform(shape=[1, 100], minval=-1, maxval=1, dtype="float32")
    target_path = load_path()
    target = load_image(get_batch_data(target_path, 13, 1), inpainting=True)
    target = target.reshape(64,64)


    context_loss, prior_loss ,code= train(100,code,target)

    plt.plot(context_loss)
    plt.title("the context_loss")
    plt.savefig("result_image_inpainting3/context_loss.jpg")
    plt.close()

    plt.plot(prior_loss)
    plt.title("the prior_loss")
    plt.savefig("result_image_inpainting3/prior_loss.jpg")
    plt.close()



