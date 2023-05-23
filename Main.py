from Adversarial import WGAN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Instantiate the WGAN model
wgan = WGAN()

# Train the WGAN model
wgan.train(epochs=100)

# Generate adversarial samples
adv_samples = wgan.generate_samples(int(len(wgan.test_data)*0.2))
adv_samps_df = pd.DataFrame(adv_samples, columns=wgan.test_data.columns.values)
unnormed_samps = wgan.input_handler.unNormalize_features(adv_samps_df, wgan.input_handler.test_norm_data)
unnormed_samps.to_csv("./Adverserial_Input.csv")
#np.savetxt('AdvSamples.csv', unnormed_sampes, delimiter=',')

# Load the black-box IDS model
ids_model = load_model('IDS_Model.h5')

# Evaluate the original dataset and the adversarial samples using the IDS model
orig_scores = ids_model.evaluate(wgan.test_data, wgan.test_data_label, verbose=0)
adv_scores = ids_model.evaluate(adv_samples, np.zeros(len(adv_samples)), verbose=0)

# Print the evaluation results
print("Original Test Set - Loss: {}, Accuracy: {}".format(orig_scores[0], orig_scores[1]))
print("Adversarial Test Set - Loss: {}, Accuracy: {}".format(adv_scores[0], adv_scores[1]))

