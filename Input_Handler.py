import pandas as pd
import numpy as np

class InputHandler():

    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_norm_data = None
        self.test_norm_data = None

    def unNormalize(self, df, df_norm):
        new_df = df.drop('malicious', axis=1)
        end_col = df['malicious']
        for i in range(0, new_df.shape[1]):
            cur_col = new_df.iloc[:,i]
            cur_std, cur_avg = df_norm.iloc[i]
            col_unorm = (cur_col * cur_std) + cur_avg            
            new_df.iloc[:,i] = col_unorm
        new_df['malicious'] = end_col
        return new_df

    def unNormalize_features(self, df, df_norm):
        new_df = df.copy()
        for i in range(0, new_df.shape[1]):
            cur_col = new_df.iloc[:,i]
            cur_std, cur_avg = df_norm.iloc[i]
            col_unorm = (cur_col * cur_std) + cur_avg            
            new_df.iloc[:,i] = col_unorm
        return new_df

    def get_data_from_csv(self, unNorm=True):
        print(f"[*] Importing data from csv...")
        train_df = pd.read_csv('./outfiles/test_train/Train_oi.csv')
        test_df = pd.read_csv('./outfiles/test_train/Test_oi.csv')
        self.train_norm_data = pd.read_csv('./outfiles/test_train/Train_Normalized.csv')
        self.test_norm_data = pd.read_csv('./outfiles/test_train/Test_Normalized.csv')
        print("[+] Finished importing from file...")
        print("[*] Beginning de-normalization...")
        if unNorm:
            test_denorm = self.unNormalize(test_df, self.test_norm_data)
            train_denorm = self.unNormalize(train_df, self.train_norm_data)
            print("[+] Finished de-norm process...")
            return test_denorm, train_denorm 
        return test_df, train_df
