import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
import tensorflow as tf

def build_model(input_shape):
    print(f"[*] Building model of shape {input_shape}")
    model = Sequential()
    model.add(Dense(10, input_dim=input_shape[0], activation='relu'))
    model.add(Dense(50, input_dim=input_shape[0], activation='relu'))
    model.add(Dense(10, input_dim=input_shape[0], activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dense(input_shape[1],activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("[+] Successfully built model")
    return model

def train_model(train_data, train_labels, test_data, test_labels, epochs=100, batch_size=64, early_stopping_patience=10):
    print(f"[*] Starting model training...")
    print(f"[*] train_data : labels - {train_data.shape} : {train_labels.shape}")
    print(f"[*] test_data : labels - {test_data.shape} : {test_labels.shape}")
    print(f"[*] Epochs: {epochs}, batch_size: {batch_size}, early_stopping_patience: {early_stopping_patience}")
    input_shape = (train_data.shape[1], test_labels.shape[1])
    print(f"[+] Using input_shape = {input_shape}...")
    model = build_model(input_shape)
    early_stopping = EarlyStopping(patience=early_stopping_patience, monitor='val_loss', restore_best_weights=True)
    print(f"[+] Expanding dimensions completed...")
    print(f"[+] train_data : labels - {train_data.shape} : {train_labels.shape}...")
    print(f"[+] test_data : labels - {test_data.shape} : {test_labels.shape}...")
    train_data_arr = np.array(train_data, dtype=np.float32)
    train_labels_arr = np.array(train_labels, dtype=np.float32)
    test_data_arr = np.array(test_data, dtype=np.float32)
    test_labels_arr = np.array(test_labels, dtype=np.float32)
    print(f"[+] Feature array shapes: {train_data_arr.shape} : {test_data_arr.shape}")
    print(f"[+] Label array shapes: {train_labels_arr.shape} : {test_labels_arr.shape}")
    print("[+] Created Numpy arrays")
    with tf.device('/device:GPU:0'):
        model.fit(train_data_arr, train_labels_arr, validation_data=(test_data_arr, test_labels_arr), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
        print("[+] Finished model fiting onto test data")
    
    probabilities = model.predict(test_data_arr)
    predicted_labels = probabilities.argmax(axis=-1)
    test_labels_argmax = test_labels_arr.argmax(axis=-1).astype(int)
    print(f"[+] probabilities: {probabilities.shape}, test_labels_arr: {test_labels_arr.shape}")
    print(f"[+] predicted_labels: {predicted_labels.shape}, test_labels_argmax: {test_labels_argmax.shape}")
    accuracy = accuracy_score(test_labels_argmax, predicted_labels)
    precision = precision_score(test_labels_argmax, predicted_labels, average='macro')
    recall = recall_score(test_labels_argmax, predicted_labels, average='macro')
    f1 = f1_score(test_labels_argmax, predicted_labels, average='macro')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 score:', f1)
    return model

def save_model(model):
    model.save("IDS_Model.h5")

def import_data():
    #Import data
    print("[*] Importing train + test data from csv file...")
    train_data = pd.read_csv('./outfiles/test_train/Train_oi.csv')
    print("[+] Imported train_data")
    test_data = pd.read_csv('./outfiles/test_train/Test_oi.csv')
    print("[+] Imported train_data")
    print(f"[+] Successfully imported training : test data - {train_data.shape} : {test_data.shape} ...")
    return (test_data, train_data)

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()
    if sd is None:
        sd = df[name].std()
    return (df[name]-mean)/sd

# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

def encode_feature_labels(train_data, test_data):
    train_labels = train_data['Label']
    train_data_features = train_data.drop('Label', axis=1)
    test_labels = test_data['Label']
    test_data_features = test_data.drop('Label', axis=1)
    feature_columns = train_data_features.columns
    print(f"[*] zscore normalization for {len(feature_columns)} columns")
    for col in feature_columns:
        train_data_features[col] = encode_numeric_zscore(train_data_features, col)
        test_data_features[col] = encode_numeric_zscore(test_data_features, col)
    print("[+] normalization complete")    
    print("[*] Starting one hot encoding")
    train_label_dummies = pd.get_dummies(train_labels) # Classification
    test_label_dummies = pd.get_dummies(test_labels) # Classification
    print("[+] One hot encoding finished")
    print(f"[+] Final feature shapes- {train_data_features.shape} : {test_data_features.shape}")
    print(f"[+] Final label shapes- {train_label_dummies.shape} : {test_label_dummies.shape}")
    return ((train_data_features, train_label_dummies), (test_data_features, test_label_dummies))

def main():
    # Import train + test data
    test_data, train_data = import_data()
    (train_data_features, train_label_dummies), (test_data_features, test_label_dummies) = encode_feature_labels(train_data, test_data)

    # split data into training and validation sets
    train_data_features, val_data_features, train_labels, val_labels = train_test_split(train_data_features, train_label_dummies, test_size=0.2, random_state=42)
    print("[+] Split train into training and validation...")

    # train and evaluate model
    model = train_model(train_data_features, train_labels, val_data_features, val_labels)
    print("[+] Finished model training")
    
    #test_data_arr = np.array(test_data, dtype=np.float32)
    test_data_features = np.array(test_data_features, dtype=np.float32)
    test_labels_arr = np.array(test_label_dummies, dtype=np.float32)
    test_labels_argmax = test_labels_arr.argmax(axis=-1).astype(int)

    probabilities = model.predict(test_data_features)
    predicted_labels = probabilities.argmax(axis=-1)
    accuracy = accuracy_score(test_labels_argmax, predicted_labels)
    print('Test accuracy:', accuracy)

    save_model(model)

if __name__ == '__main__':
    main()
