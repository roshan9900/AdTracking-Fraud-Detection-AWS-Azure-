import os
import sys
from src.exception import customException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # load dataset
            df = pd.read_csv(
                r'C:\Users\hp\Documents\AdTrackingFruadDetection\data\train_sample.csv'
            )
            logging.info(f'Dataset loaded with shape: {df.shape}')

            # save raw
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Raw dataset saved at {self.ingestion_config.raw_data_path}')

            # split
            logging.info('Train-test split initiated')
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df['is_attributed']
            )

            # save train/test
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f'Train set saved at {self.ingestion_config.train_data_path} with shape {train_set.shape}')
            logging.info(f'Test set saved at {self.ingestion_config.test_data_path} with shape {test_set.shape}')

            logging.info('Data ingestion completed successfully')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise customException(e, sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    print(f"✅ Train data at: {train_data}")
    print(f"✅ Test data at: {test_data}")
