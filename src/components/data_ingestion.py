#!kaggle competitions download -c talkingdata-adtracking-fraud-detection

# import zipfile
# import os

# zip_path = r"C:/Users/hp/Documents/AdTrackingFruadDetection/src/notebooks/talkingdata-adtracking-fraud-detection.zip"
# extract_path = r"C:/Users/hp/Documents/AdTrackingFruadDetection/data"

# # Create folder if it doesn’t exist
# os.makedirs(extract_path, exist_ok=True)

# # Extract the ZIP
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_path)

# print("✅ Extraction completed!")


import os
import sys
from src.exception import customException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(r'C:\Users\hp\Documents\AdTrackingFruadDetection\data\train_sample.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)
            logging.info('train test split initiated')
            train_set, test_set = train_test_split(df, test_size=.2, random_state=42,stratify=df['is_attributed'])
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            train_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)

            logging.info('ingestion of the data is completed')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise customException(e,sys)



if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()