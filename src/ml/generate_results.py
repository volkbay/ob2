import cv2
import tensorflow as tf
from tensorflow.keras import layers

IN_WIDTH = int(320)
IN_HEIGHT = int(240)
noise_dim = 1000

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(IN_HEIGHT/4*IN_WIDTH/4*64, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(IN_HEIGHT/4), int(IN_WIDTH/4), 64)))
    assert model.output_shape == (None, int(IN_HEIGHT/4), int(IN_WIDTH/4), 64) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(IN_HEIGHT/4), int(IN_WIDTH/4), 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(IN_HEIGHT/2), int(IN_WIDTH/2), 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, IN_HEIGHT, IN_WIDTH, 1)

    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[IN_HEIGHT, IN_WIDTH, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.1))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

test_input = tf.random.normal([50, noise_dim])
predictions = generator(test_input, training=False)

for i in range(predictions.shape[0]):
    img = predictions[i, :, :, 0] * 127.5 + 127.5
    cv2.imwrite("./result/res{}.png".format(i+1), img.numpy())