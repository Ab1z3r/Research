import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Input_Handler import InputHandler
import logging
tf.get_logger().setLevel(logging.ERROR)

class WGAN():

    def __init__(self, input_shape=(68,), noise_dim=100, gen_hidden_dim=256, dis_hidden_dim=64, learning_rate=0.0002, batch_size=64, n_critic=5):
        print("[*] Starting WGAN __init__")
        self.input_shape = input_shape
        self.noise_dim = noise_dim
        self.gen_hidden_dim = gen_hidden_dim
        self.dis_hidden_dim = dis_hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.clip_value = 0.01
        print("[+] Initialized basic vars...")
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=self.wasserstein_loss, optimizer=Adam(learning_rate, beta_1=0.5, beta_2=0.9))
        print("[+] Finished generating discriminator...")
        # Build the generator
        self.generator = self.build_generator()
        print("[+] Finished generating generator...")
        # Build the combined model
        self.combined = self.build_combined()
        print("[+] Finished generating combined...")
        # Load the data
        self.input_handler = InputHandler()
        test_data, train_data = self.input_handler.get_data_from_csv(unNorm=False)
        self.test_data = test_data.drop('malicious', axis=1)
        self.test_data_label = test_data['malicious']
        self.train_data = train_data.drop('malicious', axis=1)
        self.train_data_label = train_data['malicious']
        print("[+] Extracted data completely...")

    def wasserstein_loss(self, y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    def build_generator(self):
        input_noise = Input(shape=(self.noise_dim,))
        x = Dense(self.gen_hidden_dim, activation='relu')(input_noise)
        x = Dense(self.gen_hidden_dim, activation='relu')(x)
        x = Dense(np.prod(self.input_shape), activation='tanh')(x)
        output = Reshape(self.input_shape)(x)
        model = Model(input_noise, output)
        return model

    def build_discriminator(self):
        input_data = Input(shape=self.input_shape)
        x = Flatten()(input_data)
        x = Dense(self.dis_hidden_dim)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(self.dis_hidden_dim)(x)
        x = LeakyReLU(alpha=0.2)(x)
        output = Dense(1)(x)
        model = Model(input_data, output)
        return model

    def build_combined(self):
        input_noise = Input(shape=(self.noise_dim,))
        generated_data = self.generator(input_noise)
        validity = self.discriminator(generated_data)
        model = Model(input_noise, validity)
        model.compile(loss=self.wasserstein_loss, optimizer=Adam(self.learning_rate, beta_1=0.5, beta_2=0.9))
        return model

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.noise_dim))
        generated_samples = self.generator.predict(noise)
        return generated_samples

    def generate_noise(self, n_samples):
        return np.random.normal(0, 1, (n_samples, self.noise_dim))

    def train(self, epochs=5000):
        print("[*] Starting Training...")
#        self.train_data = np.array(self.train_data)
#        self.train_data = np.reshape(self.train_data, (-1, self.input_shape[0]))
        for epoch in range(epochs):
            # Train the discriminator for n_critic iterations
            for _ in range(self.n_critic):
                # Select a random batch of data
                idx = np.random.randint(0, self.train_data.shape[0], self.batch_size)
                real_data = self.train_data.iloc[idx]
                # Generate a batch of fake data
                noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
                fake_data = self.generator.predict(noise)
                # Clip the weights of the discriminator
                for layer in self.discriminator.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    layer.set_weights(weights)
                # Train the discriminator on real and fake data
                d_loss_real = self.discriminator.train_on_batch(real_data, -np.ones((self.batch_size, 1)))
                d_loss_fake = self.discriminator.train_on_batch(fake_data, np.ones((self.batch_size, 1)))
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            self.discriminator.trainable = False
            # Train generator
            gan_input = self.generate_noise(len(self.train_data))
            y_gen = -np.ones((len(self.train_data), 1))
            g_loss = self.combined.train_on_batch(gan_input, y_gen)
            # Print progress
            print("[+] Epoch: %d  Discriminator Loss: %f  Generator Loss: %f" % (epoch, d_loss, g_loss))
            # Save generator every 500 epochs
#            if epoch % 500 == 0:
#                self.save_model(epoch)
