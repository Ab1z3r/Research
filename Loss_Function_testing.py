import tensorflow as tf
import tensorflow.keras as layers
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from IDS_Model import IDS_Model

class Adversarial_Try:
    def __init__(self, k_steps=10, epochs=1, batch_size=64, lambda_val=0.5):
        print(f"[*] Initializing attack: epochs-{epochs}, batch_size:{batch_size}, lambda_val:{lambda_val}")
        self.epochs = epochs
        self.k_steps = k_steps
        self.batch_size = batch_size
        self.lambda_val = lambda_val
        self.gen_model = self.build_generator()
        self.disc_model = self.build_discriminator()
        self.ids_model = IDS_Model()
        self.ids_model.load_ids()
        return


    def build_generator(self, input_dim=(68,), output_features=68):
        print("[*] Building generator...")
        model = tf.keras.Sequential()
        # layer 1
        model.add(Dense(128, input_shape=(input_dim)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # layer 2
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # layer 3
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # layer 4
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        #layer 5
        model.add(Dense(output_features, activation='tanh'))
        print("[+] Generator built...")
        return model

    def build_discriminator(self, input_dim=(68,), output_features=1):
        print("[*] Building discriminator...")
        model = tf.keras.Sequential()
        # layer 1
        model.add(Dense(128, input_shape=(input_dim)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # layer 2
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # layer 3
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # layer 4
        model.add(Dense(output_features, activation='tanh'))
        print("[+] Discriminator built...")
        return model

    def main_algo(self):
        for epoch in range(0, self.epochs):
            print("[*] Running epoch {epoch}")
            for iteration in range(0, self.k_steps):
                print("[*] Running iteration {iteration}...")
                # Extract m Normal samples
                normal_data = None
                # Extract m Malicious samples
                mal_data = None
                # Applying loss function
                ## D(x)
                D_normal = self.disc_model(normal_data)
                ## D(z)
                D_mal = sef.disc_model(mal_data)
                ## loss func
                loss_discrim = tf.reduce_mean(D_normal - D_mal)
                ## Get trainable vars
                discrim_vars = self.disc_model.trainable_variables
                ## Compute Grad wrt trainable vars
                gradients = tf.gradients(loss_discrim, discrim_vars)
                ## Define optimizer for updating vars
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                ## Apply gradient to update trainable vars
                train_op_discrim = optimizer.apply_gradient(zip(gradients, discrim_vars))
                ## Set clipping threshold [-c,c]?
                print("[+] Finished running iteration {iteration}...")
                # END FOR
            ## Ectract m malicious samples
            mal_data_gen = None
            ## Calc discrim output
            D_generated = self.disc_model(mal_data_gen)
            ## calculate IDS model classification
            mal_data_gen = np.array(mal_data_gen, dtype=np.float32)
            IDS_Generated = self.ids_model.ids_model(mal_data_gen).numpy()
            predicted_labels = np.array(IDS_Generated.argmax(axis=-1), dtype=np.int64)
            target_labels = np.array(np.zeros(self.batch_size), dtype=np.int64)
            ## Define adv loss
            adv_loss = -tf.reduce_mean(D_generated)
            ## Define IDS loss based on predefined loss func
            ids_loss = self.ids_model.IDS_loss(predicted_labels, target_labels)
            ## Define final generator loss
            generator_loss = adv_loss + self.lambda_val * ids_loss
            ## Extract generators training variables
            generator_variables = self.gen_model.trainable_variables
            ## Compute gradient wrt loss func
            gradients = tf.gradients(generator_loss, generator_variables)
            ## Define optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            ## Apply gradient descent
            train_op_genr = optimizer.apply_gradient(zip(gradients, generator_variables))
            print("[+] Finished running epoch {epoch}...")
            #END FOR
