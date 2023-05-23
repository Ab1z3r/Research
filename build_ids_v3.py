import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

def build_model(input_shape):
    print(f"[*] Building model of shape {input_shape}")
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    print("[+] Successfully built model")
    return model

def train_model(train_data, train_labels, test_data, test_labels, epochs=100, batch_size=64, early_stopping_patience=10):
    print(f"[*] Starting model training...")
    print(f"[*] train_data : labels - {train_data.shape} : {train_labels.shape}")
    print(f"[*] test_data : labels - {test_data.shape} : {test_labels.shape}")
    print(f"[*] Epochs: {epochs}, batch_size: {batch_size}, early_stopping_patience: {early_stopping_patience}")

    input_shape = (train_data.shape[1], 1)
    print(f"[+] Using input_shape = {input_shape}...")

    model = build_model(input_shape)

    early_stopping = EarlyStopping(patience=early_stopping_patience, monitor='val_loss', restore_best_weights=True)
    #train_data = np.array(train_data, axis=2)
    #test_data = np.array(test_data, axis=2)
    print(f"[+] Expanding dimensions completed...")
    print(f"[+] train_data : labels - {train_data.shape} : {train_labels.shape}...")
    print(f"[+] test_data : labels - {test_data.shape} : {test_labels.shape}...")

    train_data_arr = np.array(train_data, dtype=np.float32)
    train_labels_arr = np.array(train_labels, dtype=np.float32)
    test_data_arr = np.array(test_data, dtype=np.float32)
    test_labels_arr = np.array(test_labels, dtype=np.float32)

    print("[+] Created Numpy arrays")

    model.fit(train_data_arr, train_labels_arr, validation_data=(test_data_arr, test_labels_arr), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    print("[+] Finished model fiting onto test data")

    probabilities = model.predict(test_data_arr)
    predicted_labels = probabilities.argmax(axis=-1)
    accuracy = accuracy_score(test_labels_arr, predicted_labels)
    precision = precision_score(test_labels_arr, predicted_labels)
    recall = recall_score(test_labels_arr, predicted_labels)
    f1 = f1_score(test_labels_arr, predicted_labels)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    return model

def save_model(model):
    model.save("IDS_Model.h5")

def main():

    #Import data
    print("[*] Importing train + test data from csv file...")
    train_data = pd.read_csv('./outfiles/test_train/Train_oi.csv')
    test_data = pd.read_csv('./outfiles/test_train/Test_oi.csv')
    df_norm_vars = pd.read_csv('./outfiles/test_train/Normalized.csv')
    print(f"[+] Successfully imported training : test data - {train_data.shape} : {test_data.shape} : {df_norm_vars.shape}...")

    # extract labels
    train_labels = train_data['malicious']
    test_labels = test_data['malicious']

    # encode labels as integers
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_labels)
    test_labels = label_encoder.transform(test_labels)

    print("[+] Extracted and encoded labels...")

    # drop label column from data
    train_data = train_data.drop('malicious', axis=1)
    test_data = test_data.drop('malicious', axis=1)

    print("[+] Extracted {train_data.shape[1]} features from data...")

    # split data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    print("[+] Split train into training and validation...")

    # train and evaluate model

    model = train_model(train_data, train_labels, val_data, val_labels)
    print("[+] Finished model training")
    
    test_data_arr = np.array(test_data, dtype=np.float32)
#    test_data = np.expand_dims(test_data, axis=2)
    probabilities = model.predict(test_data_arr)
    predicted_labels = probabilities.argmax(axis=-1)
#    predicted_labels = model.predict_classes(test_data)
    accuracy = accuracy_score(test_labels, predicted_labels)
    print('Test accuracy:', accuracy)
    save_model(model)

if __name__ == '__main__':
    main()
