from typing import List
import pandas as pd 
import numpy as np



class DataHandler:
    def __init__(self, features_names: str, test_folder_pth: str, train_folder_pth: str) -> None:
        self.features_names=features_names
        self.column_names=self.get_columns()
        self.test_folder=test_folder_pth
        self.train_folder=train_folder_pth
        self.df_train=self.train_data()
        self.df_test=self.test_data()
        
    def load_data(self,file_path: str)-> pd.DataFrame:
        df=pd.read_table(file_path, sep=' ', header=None)
        return df

    def get_columns(self):
        pass
    
    def train_data(self):
        x_train=self.load_data(file_path=self.train_folder+'/X_train.txt')
        y_train=self.load_data(file_path=self.train_folder+'/y_train.txt')
        df_train=pd.concat([x_train, y_train], axis=1)
        df_train.columns=self.column_names

        return df_train
    
    def test_data(self):
        x_test=self.load_data(file_path=self.train_folder+'/X_test.txt')
        y_test=self.load_data(file_path=self.train_folder+'/y_test.txt')
        df_test=pd.concat([x_test, y_test], axis=1)
        df_test.columns=self.column_names

        return df_test
    
    
    def df_train_batch(self, batch_size, n_steps,n_features):
        
        t_max = self.df_train.shape[0]
    
        
        x = np.zeros((batch_size,n_steps,n_features))
    
        y = np.zeros((batch_size,n_steps,))

        

        starting_points = np.random.randint(0,t_max-n_steps-1,size=batch_size)    
        #print(starting_points)
        
        feat = self.df_train.values

        # We create the batches for x using all time series (8) between t and t+n_steps    
        for i, sp in enumerate(starting_points):
            x[i] = feat[sp:sp+n_steps]
            y[i] = feat[sp+1:sp+n_steps+1, 1]
            
        # We create the batches for y using only one time series between t+1 and t+n_steps+1
        
        #Save on x and y the time series data sequence and the prediction sequence

        return x,y



# df=pd.read_table('E:/Datasets/HAPT Data Set/Test/y_test.txt', sep=' ', header=None)

# #print(df.shape)
# dft=[]
# with open('E:/Datasets/HAPT Data Set/features.txt', ) as f:
#     print(f.read())
    



        