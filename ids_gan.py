import os
import gc
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import sklearn
import imblearn
import matplotlib.pyplot as plt
import time
import sklearn.metrics as m
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# Class designed to modify the dataset
class InputHandler():

    # Costructor for input handler
    def __init__(self):
        print(f"[*] Called init for InputHandler...")
        self.clean()
    
    # Clean the garbage pipe
    def clean(self):
        return gc.collect()
    
    # Check data set file for integrity
    def check_data(self):
        print("[*] Beginning walk of input directory for CIC-IDS2017 dataset\n Please verify all files are present...")
        for dirname, _, filenames in os.walk('/input'):
            for filename in filenames:
                print(os.path.join(dirname, filename))
        print("[+] End Code check_data()!!!")

    # Wrapper function to preprocess input
    def preprocess_input(self):
        print("[*] Beginning data import...")
        df = self.concatenate_input()
        print(f"[+] Successfully extracted input data with shape {df.shape}...")
        df = self.clean_input_features(df)
        print(f"[+] Imported data with {(df.shape[0], df.shape[1])}!!!")
        return df
    
    # Function to extract entire dataset and concatenate
    def concatenate_input(self):
        print(f"[*] Starting input concatenation...")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(f"[*] Beginning input data read...")
        df1=pd.read_csv("./input/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
        print("[+] finished reading df 1...")
        df2=pd.read_csv("./input/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
        print("[+] finished reading df 2...")
        df3=pd.read_csv("./input/Friday-WorkingHours-Morning.pcap_ISCX.csv")
        print("[+] finished reading df 3...")
        df4=pd.read_csv("./input/Monday-WorkingHours.pcap_ISCX.csv")
        print("[+] finished reading df 4...")
        df5=pd.read_csv("./input/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
        print("[+] finished reading df 5...")
        df6=pd.read_csv("./input/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
        print("[+] finished reading df 6...")
        df7=pd.read_csv("./input/Tuesday-WorkingHours.pcap_ISCX.csv")
        print("[+] finished reading df 7...")
        df8=pd.read_csv("./input/Wednesday-workingHours.pcap_ISCX.csv")
        print("[+] finished reading df 8...")
        print("[+] Finished importing entire dataset!!!")
        df = pd.concat([df1,df2])
        del df1,df2
        df = pd.concat([df,df3])
        del df3
        df = pd.concat([df,df4])
        del df4
        df = pd.concat([df,df5])
        del df5
        df = pd.concat([df,df6])
        del df6
        df = pd.concat([df,df7])
        del df7
        df = pd.concat([df,df8])
        del df8
        print(f"[+] Finished dataset concatenatenation with df of shape {df.shape}")
        return df

    # Clean the input features from the dataframes, returns [68,]
    def clean_input_features(self, df):
        print("[*] Starting Input Fleature trimming...")
        for i in df.columns:
            df = df[df[i] != "Infinity"]
            df = df[df[i] != np.nan]
            df = df[df[i] != ",,"]
        df[['Flow Bytes/s', ' Flow Packets/s']] = df[['Flow Bytes/s', ' Flow Packets/s']].apply(pd.to_numeric)
        df.drop([' Bwd PSH Flags'], axis=1, inplace=True)
        df.drop([' Bwd URG Flags'], axis=1, inplace=True)
        df.drop(['Fwd Avg Bytes/Bulk'], axis=1, inplace=True)
        df.drop([' Fwd Avg Packets/Bulk'], axis=1, inplace=True)
        df.drop([' Fwd Avg Bulk Rate'], axis=1, inplace=True)
        df.drop([' Bwd Avg Bytes/Bulk'], axis=1, inplace=True)
        df.drop([' Bwd Avg Packets/Bulk'], axis=1, inplace=True)
        df.drop(['Bwd Avg Bulk Rate'], axis=1, inplace=True)
        df.drop(['Flow Bytes/s',' Flow Packets/s'], axis=1, inplace=True)
        # Keep only 5 attacks and 1 benign class
        # 0 = BENIGN; 1 = slowloris; 2 = heartbleed; 3 = sql_injection; 4 = goldeneye; 5 = ssh_patator
        rows_to_keep = ['BENIGN', 'DoS slowloris', 'Heartbleed', 'Web Attack � Sql Injection', 'DoS GoldenEye', 'SSH-Patator']
        mask = df[' Label'].isin(rows_to_keep)
        df = df[mask]
        new_names = [0, 1, 2, 3, 4, 5]
#        new_names = ['benign', 'slowloris', 'heartbleed', 'sql_injection', 'goldeneye', 'ssh_patator']
        rename_dict = {rows_to_keep[i]: new_names[i] for i in range(len(rows_to_keep))}
        df = df.rename(columns={' Label': 'Label'})
        df['Label'] = df['Label'].replace(rename_dict)
        print(f"[+] Finished feature trimming, final df shape {df.shape}")
        return df

    def normalize_data(self, test_data, train_data):
        test_data_labels = test_data['Label']
        train_data_labels = train_data['Label']
        train_data = train_data.drop('Label', axis=1)
        test_data = test_data.drop('Label', axis=1)

        test_data_num = test_data.select_dtypes(include='number')
        test_data_norm = (test_data_num - test_data_num.mean()) / test_data_num.std()
        test_data_num_max_min = pd.DataFrame({'std': test_data_num.std(), 'mean': test_data_num.mean()})
        test_data[test_data_norm.columns] = test_data_norm
        test_data['Label'] = test_data_labels

        train_data_num = train_data.select_dtypes(include='number')
        train_data_norm = (train_data_num - train_data_num.mean()) / train_data_num.std()
        train_data_num_max_min = pd.DataFrame({'std': train_data_num.std(), 'mean': train_data_num.mean()})
        train_data[train_data_norm.columns] = train_data_norm
        train_data['Label'] = train_data_labels

        return (test_data, test_data_num_max_min), (train_data, train_data_num_max_min)

    # Main function to orchestrate module
    def main_func(self):    
        self.clean()
        df = self.preprocess_input()
        print(f"[+] Successfully imported dataset!!!")
        train, test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
        print(f"[+] Train - Test split success: {train.shape} : {test.shape}")
        train.to_csv('./outfiles/test_train/Train_oi.csv',index=False)
        test.to_csv('./outfiles/test_train/Test_oi.csv',index=False)
        print(f"[+] Finished writing both training and test data into csv")
        print(f"[*] test shape : train shape = {test.shape} : {train.shape}")
        print(f"[*] Here are some stats about the train data:")
        print(train.iloc[:,-1:].value_counts())
        print(f"[*] Here are some stats about the test data:")
        print(test.iloc[:,-1:].value_counts())
        print("[+] Here are some stats about the imported dataset:")
        print(df.iloc[:,-1:].value_counts())
        self.clean()
        del train
        del test
        print("[+] Deleted main test and train data...")
        return
