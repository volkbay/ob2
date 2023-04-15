import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import time
import glob
import numpy as np
import cv2

IN_WIDTH = int(320)
IN_HEIGHT = int(240)
BUFFER_SIZE = 1000
BATCH_SIZE = 10
EPOCHS = 50
LEARN_RATE = 1e-4
CONV_SIZE = (3, 3)
START_SHAPE = 61
RANGE_SHAPE = 2
NEG_IMAGES = 5

FOLDER = './crop{}/*'.format(str(IN_HEIGHT))
input_list = [  1,    8,   18,   29,   44,   59,   71,   84,   99,  113,
              121,  133,  144,  154,  164,  175,  186,  198,  211,  224,
              235,  245,  256,  262,  272,  281,  293,  301,  309,  319,  329,
              337,  345,  347,  361,  371,  386,  396,  411,  421,  430,
              439,  450,  458,  477,  485,  502,  517,  524,  535,  554,
              574,  592,  601,  616,  631,  641,  652,  666,  679,  693,
              703,  713,  728,  738,  751,  767,  782,  796,  811,  821,
              832,  841,  849,  861,  871,  880,  891,  901,  912,  922,
              933,  944,  957,  966,  980,  991, 1007, 1021, 1031, 1043,
             1053, 1061, 1071, 1081, 1091, 1101, 1111, 1121, 1131]

noise_dim = 1000
num_examples_to_generate = 20

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(IN_HEIGHT*IN_WIDTH*4, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(IN_HEIGHT/4), int(IN_WIDTH/4), 64)))
    assert model.output_shape == (None, int(IN_HEIGHT/4), int(IN_WIDTH/4), 64) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(32, CONV_SIZE, strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(IN_HEIGHT/4), int(IN_WIDTH/4), 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, CONV_SIZE, strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(IN_HEIGHT/2), int(IN_WIDTH/2), 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, CONV_SIZE, strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, IN_HEIGHT, IN_WIDTH, 1)

    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, CONV_SIZE, strides=(1, 1), padding='same',
                                     input_shape=[IN_HEIGHT, IN_WIDTH, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, CONV_SIZE, strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, CONV_SIZE, strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


generator = make_generator_model()
generator.save_weights('gen.h5')
discriminator = make_discriminator_model()
discriminator.save_weights('disc.h5')

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(LEARN_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARN_RATE)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, epochs, shape_no):
  test_input = tf.random.normal([4, noise_dim])
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)

    if (epoch + 1) % 5 == 0:
        predictions = generator(test_input, training=False)
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(2, 2, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title("Gen Loss: {:0.3f} Disc Loss: {:0.3f}".format(gen_loss, disc_loss), loc='left')
        plt.savefig('./result/image_{}_epoch_{:04d}.png'.format(shape_no, epoch+1))
        plt.close()

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print('Last gen loss: {} and disc loss: {}'.format(gen_loss, disc_loss))


def generate_and_save_images(model, test_input, shape_no):
  predictions = model(test_input, training=False)

  for i in range(predictions.shape[0]):
      img = predictions[i, :, :, 0] * 127.5 + 127.5
      cv2.imwrite("./result/res{}_{}.png".format(shape_no, i+1), img.numpy())


# MAIN FUNCTION #
for m, n in enumerate(input_list):
    if m >= START_SHAPE:
        IMAGE_COUNT = input_list[m+1] - n + NEG_IMAGES
        train_images = np.zeros((IMAGE_COUNT, IN_HEIGHT, IN_WIDTH, 1))
        imList = glob.glob(FOLDER)
        for file in imList:
            str = os.path.split(file)
            index = str[-1][7:-4]  # Take file number
            if int(index) in range(n, input_list[m+1]):
                train_images[int(index)-n, :, :, 0] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        negList = glob.glob("./neg/*.png")
        negList = np.random.choice(negList, 1, False)
        """for dummy, file in enumerate(negList):
            train_images[IMAGE_COUNT-NEG_IMAGES+dummy, :, :, 0] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)"""
        for dummy in range(NEG_IMAGES):
            print(negList[0])
            train_images[IMAGE_COUNT - NEG_IMAGES + dummy, :, :, 0] = cv2.imread(negList[0], cv2.IMREAD_GRAYSCALE)

        train_images = np.tile(train_images, (np.ceil(BUFFER_SIZE / IMAGE_COUNT).astype('int'), 1, 1, 1))
        train_images = train_images.reshape((train_images.shape[0], IN_HEIGHT, IN_WIDTH, 1)).astype('float32')
        train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images[0:BUFFER_SIZE]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

        time_train = time.time()
        train(train_dataset, EPOCHS, m+1)
        print("Total Training Time: {:0.3f} min \n".format((time.time()-time_train) / 60.0))

        seed = tf.random.normal([num_examples_to_generate, noise_dim])
        generate_and_save_images(generator, seed, m+1)

        generator.load_weights("gen.h5")
        discriminator.load_weights("disc.h5")

        if m+1 == START_SHAPE+RANGE_SHAPE:
            break