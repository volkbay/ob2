import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import time
import glob
import numpy as np
import cv2
from IPython import display

IN_WIDTH = int(320)
IN_HEIGHT = int(240)
BUFFER_SIZE = 950
BATCH_SIZE = 5
EPOCHS = 75
FOLDER = './crop{}_main/*'.format(str(IN_HEIGHT))
#input_list = ['1','2','3','4','5','6','7','8','9','10']

noise_dim = 1000
num_examples_to_generate = 16
IMAGE_COUNT = 95 # len(input_list)

train_images = np.zeros((IMAGE_COUNT, IN_HEIGHT, IN_WIDTH, 1))
imList = glob.glob(FOLDER)
for i, file in enumerate(imList):
    str = os.path.split(file)
    index = str[-1][7:-4]  # Take file number
    #if index in ['1','2','3','4','5','6','7','8','9','10']:
    #    train_images[int(index)-1, :, :, 0] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    train_images[i, :, :, 0] = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

train_images = np.repeat(train_images, 10, axis=0)
train_images = train_images.reshape((BUFFER_SIZE, IN_HEIGHT, IN_WIDTH, 1)).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
"""for image_batch in train_dataset:
    print(image_batch.shape)
    #plt.imshow(image_batch[0,:,:,0], cmap='gray')
    plt.show()
"""
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

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
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

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      gen_loss, disc_loss = train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    """
    # Save the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)"""

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print('Last gen loss: {} and disc loss: {}'.format(gen_loss, disc_loss))

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
      img = predictions[i, :, :, 0] * 127.5 + 127.5
      if epoch == EPOCHS:
        cv2.imwrite("./result/res{}.png".format(i+1), img.numpy())

  plt.savefig('./result/image_at_epoch_{:04d}.png'.format(epoch))
  plt.close()

# MAIN FUNCTION
#generator.summary()
train(train_dataset, EPOCHS)
checkpoint.save(file_prefix = checkpoint_prefix)