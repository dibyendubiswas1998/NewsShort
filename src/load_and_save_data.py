import glob
from typing import Any
import pandas as pd
import os
import gzip
from ensure import ensure_annotations
from src.utils.logging import log
from src.utils.common_utils import save_raw_local_df, create_dir, clean_prev_dirs_if_exis
from src.config.configuration import ConfigurationManager
import datasets
from datasets import DatasetDict, Dataset



class DataLoader:
    """
    A class for loading and saving data from multiple CSV files.

    This class provides methods to load data from multiple CSV files in a directory and concatenate them into a single data object. It also provides functionality to save the data to a specified directory and log the status of the operation.

    Methods:
        load_csv_data(directory_path, log_file): Load data from multiple CSV files in a directory and concatenate them into a single data object.
        save_data(new_data_path, data, log_file): Save the provided data to the specified directory and log the status of the operation.
    """

    def __init__(self):
        self.config_manager = ConfigurationManager()    # load ConfigurationManager class
        self.directory_path_ = self.config_manager.get_raw_data_ingestion_config() # load the directory path from confg.yaml file
        self.training_log_file, self.prediction_log_file = self.config_manager.logs_files_config() # load training log file from confg.yaml file
        self.new_raw_data_dir, self.new_data_file_name = self.config_manager.saving_data_ingestion_config() # load new_raw_data_dir name and raw_data_file_name from confg.yaml file
        self.hugging_face_dataset = self.config_manager.hugging_face_dataset_config() # load data from hugging_face_dataset
    
        

    @ensure_annotations
    def load_csv_data(self, directory_path=None, log_file=None) -> pd.DataFrame:
        """
        Load data from multiple CSV files in a directory and concatenate them into a single data object.

        Args:
            directory_path (str): The path to the directory containing the CSV files.
            log_file (file object): The log file to which the function will write its logs.

        Returns:
            pd.DataFrame: The concatenated data from all CSV files in the directory.

        Raises:
            FileNotFoundError: If the specified directory or CSV file is not found.
            PermissionError: If there is a permission error while accessing the directory or CSV file.
        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file

            if directory_path is None:
                # If directory_path is not provided, use the default value from ConfigurationManager()
                directory_path = self.directory_path_

            all_data = []  # List to store all data frames
            # Get a list of all gzip-compressed CSV files in the directory
            csv_files = glob.glob(os.path.join(directory_path, '*.csv.gz'))

            # Iterate through each gzip-compressed CSV file, unzip it, and load its data into a DataFrame
            for csv_file in csv_files:
                csv_path = csv_file
                with gzip.open(csv_path, 'rt', encoding='utf-8') as gz_file:
                    data_frame = pd.read_csv(gz_file)
                    all_data.append(data_frame)
                    log(file_object=log_file, log_message=f"successfully read {csv_file} from {directory_path} and shape of the data: {data_frame.shape}")  # logs the details about the data

            # Concatenate all data frames into a single data object
            single_data_object = pd.concat(all_data, ignore_index=True)
            log(file_object=log_file, log_message=f"successfully read all the data from {directory_path} and shape of the data: {single_data_object.shape}")  # logs the details about the data is load successfully
            return single_data_object # returns the whole data

        except (FileNotFoundError, PermissionError) as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file

            if directory_path is None:
                # If directory_path is not provided, use the default value from ConfigurationManager()
                directory_path = self.directory_path_

            log(file_object=log_file, log_message=f"Error will be raised: {ex}")  # logs the error message
            raise ex



    @ensure_annotations
    def save_data(self, data, new_raw_data_dir=None, new_data_file_name=None, log_file=None) -> object:
        """
        Save the provided data to the specified directory and log the status of the operation.

        Args:
            data (object): The data to be saved.
            new_raw_data_dir (str, optional): The path to the directory where the data will be saved. If not provided, the default value from ConfigurationManager will be used.
            new_data_file_name (str, optional): The name of the file where the data will be saved. If not provided, the default value from ConfigurationManager will be used.
            log_file (file object, optional): The log file where the status of the operation will be recorded. If not provided, the default value from ConfigurationManager will be used.

        Returns:
            object: The saved data.

        Raises:
            ValueError: If the provided directory path is invalid or if the provided data is invalid.
            Exception: If an error occurs during the operation.
        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file

            if new_raw_data_dir is None:
                # If new_raw_data_dir is not provided, use the default value from ConfigurationManager()
                new_raw_data_dir = self.new_raw_data_dir

            if new_data_file_name is None:
                # If new_data_file_name is not provided, use the default value from ConfigurationManager()
                new_data_file_name = self.new_data_file_name

            # if not new_data_file_name or not os.path.isdir(new_data_file_name):
            #     log(file_object=log_file, log_message=f"invalid new data path") # logs the details of the error
            #     raise ValueError(f'invalid new_data_path {new_data_file_name}') # raises the exception
        
            # if not data:
            #     log(file_object=log_file, log_message=f"invalid data") # logs the details of the error
            #     raise ValueError('invalid data') # raises the exception

            clean_prev_dirs_if_exis(dir_path=new_raw_data_dir) # deletes the directory if it exists
            log(file_object=log_file, log_message=f"delete the directory if it is already exists {new_raw_data_dir}") # logs the details about the directory deletion if it exists
            
            create_dir(dirs=[new_raw_data_dir]) # create directory
            log(file_object=log_file, log_message=f"create raw data directory: {new_raw_data_dir}") # logs about directory creation

            save_raw_local_df(data=data, data_path=new_data_file_name) # svae the data directory
            log(file_object=log_file, log_message=f"data is stored in the {new_data_file_name} directory") # logs the details
    
        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            log(file_object=log_file, log_message=f"Error occurred: {ex}") # logs the error message



    @ensure_annotations
    def load_data_from_hugging_face(self, file_name=None, log_file=None) -> datasets.dataset_dict.DatasetDict:
        """
        Loads a pre-defined dataset from the Hugging Face library and returns it as a DatasetDict object.

        Args:
            file_name (str, optional): The name of the pre-defined dataset to be loaded. If not provided, the default value from the ConfigurationManager will be used.
            log_file (str, optional): The log file where the status of the operation will be recorded. If not provided, the default value from the ConfigurationManager will be used.

        Returns:
            datasets.dataset_dict.DatasetDict: The loaded pre-defined dataset.

        Raises:
            Exception: If an error occurs during the loading process.

        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
        
            if file_name is None:
                # If file_name is not provided, use the default value from ConfigurationManager()
                file_name = self.hugging_face_dataset

            data = datasets.load_dataset(file_name) # load the pre-defined dataset
            log(file_object=log_file, log_message=f"load the data from hugging face: {data}")
            return data # return the dataset

        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            log(file_object=log_file, log_message=f"Error occurred: {ex}") # logs the error message





if __name__ == "__main__":
   data_loader = DataLoader()
   data = data_loader.load_csv_data()
   data_loader.save_data(data=data)




