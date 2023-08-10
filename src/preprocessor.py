import os
from typing import Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from src.utils.logging import log
from src.config.configuration import ConfigurationManager
from ensure import ensure_annotations
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
nltk.download('punkt')
nltk.download('stopwords')

        



class PreProcessor:
    """
        PreProcessor class is responsible for preprocessing the input data for natural language processing (NLP) tasks. It provides methods to handle data, separate X and Y features, preprocess text data, perform word embedding, and split the dataset into train, test, and validation datasets.

        Methods:
        - handle_data(data=None, log_file=None) -> pd.DataFrame:
            Preprocesses the input data by filling null values using forward-fill method and logs the details about how to handle the data.

        - separate_x_y(data=None, log_file=None, input_col_name=None, output_col_name=None):
            Separates the input data into X and Y features.

        - text_preprocessing(data, log_file=None):
            Preprocesses the input text data by applying various text processing techniques.

        - word_embedding(X, Y, log_file=None, model_name=None, tokenizer_model_dir_path=None):
            Tokenizes the input data using a pre-trained tokenizer from the Hugging Face library.

        - split_and_create_tensor_(tokenized_X, tokenized_Y, random_state=None, split_ratio=None, log_file=None):
            Splits the tokenized input and output features into training, validation, and test datasets. Converts the split data into TensorFlow tensors and creates datasets for each split.

        Fields:
            - config_manager: An instance of the ConfigurationManager class.
            - training_log_file: The log file for training.
            - prediction_log_file: The log file for prediction.
            - data_path: The path to the input data.
            - output_col: The name of the output column in the input data.
            - input_col: The name of the input column in the input data.
            - split_ratio: The ratio for splitting the dataset into train, test, and validation datasets.
            - random_state: The random state for splitting the dataset.
            - default_model_ft: The default model for fine-tuning.
            - tokenizer_model_dir_path: The directory path for saving the tokenizer model.
            - summarize_model_dir_path: The directory path for saving the summarize model.
        """

    def __init__(self):
        self.config_manager = ConfigurationManager()    # load ConfigurationManager class
        self.training_log_file, self.prediction_log_file = self.config_manager.logs_files_config() # load training log file from confg.yaml file
        self.data_path = self.config_manager.get_raw_data_from_artifacts() # load the data path from artifacts directory
        self.output_col, self.input_col = self.config_manager.get_output_col_name_config() # load the output feature name
        self.split_ratio, self.random_state = self.config_manager.random_split_config() # load random state and split ratio 
        self.default_model_ft = self.config_manager.load_hugging_face_model_config() #  load default model for fine-tuning
        self.tokenizer_model_dir_path, self.summarize_model_dir_path, self.model_logs_dir_path = self.config_manager.save_model_path_config() # load tokenizer model directory path and summarize model directory path



    @ensure_annotations
    def handle_data(self, data=None, log_file=None) -> pd.DataFrame:
        """
        Preprocesses the input data by filling the null values using forward-fill method and logs the details about how to handle the data.

        Args:
            data (pandas.DataFrame, optional): A pandas dataframe containing the input data. If not provided, the default data path will be read as a pandas dataframe.
            log_file (file object, optional): A log file object to log the details. If not provided, the default log file from ConfigurationManager will be used.

        Raises:
            FileNotFoundError: If the log file is not found.
            PermissionError: If there is a permission issue with the log file.

        Returns:
            None
        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
        
            if data is None:
                # if data object is not provided, use the default data path read as pandas dataframe
                data = pd.read_csv(self.data_path)

            data.fillna(method='ffill', inplace=True) # fill the null values usng forward-fill method
            log(file_object=log_file, log_message=f"handle the data by using forward-fill method") # log the details about how to handle the data
            return data # return the data

        except (FileNotFoundError, PermissionError) as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            log(file_object=log_file, log_message=f"Error will be raised: {ex}") # logs the error message
            raise ex



    @ensure_annotations
    def separate_x_y(self, data=None, log_file=None, input_col_name=None, output_col_name=None):
        """
        Separates the input data into X and Y features.

        Args:
            data (pandas DataFrame, optional): The input data. If not provided, the default data path will be used.
            log_file (file object, optional): The log file object to log the details. If not provided, the default log file will be used.
            output_col_name (str, optional): The name of the output column in the input data. If not provided, the default output column name will be used.

        Returns:
            tuple: A tuple containing the separated X and Y features.

        Raises:
            Exception: If an error occurs during the separation process.
        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file

            if data is None:
                # if data object is not provided, use the default data path read as pandas dataframe
                data = pd.read_csv(self.data_path)

            if output_col_name is None:
                # if output_col_name is not provided, use the default output_col_name
                output_col_name = self.output_col
            
            if input_col_name is None:
                # if input_col_name is not provided, use the default input_col_name
                input_col_name = self.input_col

            # Check if the specified output_col_name is in the DataFrame columns
            if output_col_name not in data.columns:
                raise KeyError(f"Column '{output_col_name}' not found in the DataFrame.")
            
            else:
                Y = data[output_col_name] # get the Y, where type is Series
                X = data.drop(output_col_name, axis=1) # get the X
                X = X[input_col_name] # get the X where type is Series
                log(file_object=log_file, log_message="Separate the X and Y features from data.")  # logs the details about the features separations
                return X , Y  # return the X and Y features

        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file

            log(file_object=log_file, log_message=f"Error will be raised: {ex}")  # logs the error message
            raise ex




    @ensure_annotations
    def text_preprocessing(self, data, log_file=None):
        """
        Preprocesses the input text data by applying various text processing techniques.

        Args:
            data (list): The input text data to be preprocessed.
            log_file (file object, optional): A log file object to log the details. 
                If not provided, the default log file from ConfigurationManager will be used.

        Returns:
            list: A list of preprocessed sentences.

        Raises:
            Exception: If an error occurs during text processing.

        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file

            preprocessed_sentences = [] # list of preprocessed sentences
            stop_words = set(stopwords.words('english'))
            porter = PorterStemmer()
            tokenizer = RegexpTokenizer(r'\w+')

            sentences = pd.Series(data).tolist() # list of sentences
            for sentence in sentences:
                # lowering the sentences:
                text = sentence.lower()

                # remove punctuation:
                translator = str.maketrans('', '', string.punctuation)
                text = text.translate(translator)

                # remove the numbers:
                text = re.sub(r'\d+', '', text)

                # word tokenization:
                tokens = word_tokenize(text)

                # remove the stop of words:
                words = [word for word in tokens if word not in stop_words]

                # apply stemming:
                words = [porter.stem(word) for word in words]

                # get the pre-processed sentences:
                preprocessed_sentence = ' '.join(words)
                preprocessed_sentences.append(preprocessed_sentence)

            log(file_object=log_file, log_message=f"successfully apply text processing on data") # logs the details about text processing 
            return preprocessed_sentences


        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file

            log(file_object=log_file, log_message=f"Error will be raised: {ex}")  # logs the error message
            raise ex
        




    @ensure_annotations
    def word_embedding(self, X, Y, log_file=None, model_name=None, tokenizer_model_dir_path=None):
        """
        Tokenizes the input data using a pre-trained tokenizer from the Hugging Face library.

        This method takes the input features X and output features Y and applies tokenization using a pre-trained tokenizer
        fine-tuning is used. The tokenization is applied with padding and truncation to ensure all sequences have the same
        length. The tokenized X and Y features are returned as TensorFlow tensors.

        Args:
            X (list): The input features.
            Y (list): The output features.
            log_file (file object, optional): The file object for logging details. Defaults to None.
            model_name (str, optional): The name of the pre-trained model to use for tokenization. Defaults to None.

        Returns:
            tuple: A tuple containing the tokenized X and Y features.

        Raises:
            Exception: If an error occurs during the tokenization process.
        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file

            if model_name is None:
                # If model_name is not provided, use the default model for fine tuning
                model_name = self.default_model_ft
            
            if tokenizer_model_dir_path is None:
                # If tokenizer_model_dir_path is not provided, use the default model path
                tokenizer_model_dir_path = self.tokenizer_model_dir_path

            # load the model for tokenization
            
            tokenizer = AutoTokenizer.from_pretrained(model_name) # load the tokenizer for the specific model

            # apply tokenization on X and Y
            tokenized_X = tokenizer(X, max_length=512, padding=True, truncation=True, return_tensors="pt") # apply tokenization on X feature
            tokenized_Y = tokenizer(Y, max_length=512, padding=True, truncation=True, return_tensors="pt")  # apply tokenization on Y feature
            log(file_object=log_file, log_message="apply tokenization on X and Y features") # logs the details about tokenization

            
            return tokenized_X, tokenized_Y # return the tokenized X and Y features data          

        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            log(file_object=log_file, log_message=f"Error will be raised: {ex}") # logs the error message
            raise ex




    @ensure_annotations
    def split_and_create_tensor_(self, tokenized_X, tokenized_Y, random_state=None, split_ratio=None, log_file=None):
        """
        Split the tokenized input and output features into training, validation, and test datasets.
        Convert the split data into torch tensors and create datasets for each split.

        Args:
            tokenized_X: A tokenized input feature.
            tokenized_Y: A tokenized output feature.
            random_state: An integer value to set the random seed for splitting the data. If not provided, the default value from ConfigurationManager will be used.
            split_ratio: A float value to set the ratio for splitting the data. If not provided, the default value from ConfigurationManager will be used.
            log_file: A log file object to log the details. If not provided, the default log file from ConfigurationManager will be used.

        Returns:
            train_dataset: A torch dataset containing the training data.
            validation_dataset: A torch dataset containing the validation data.
            test_dataset: A torch dataset containing the test data.
        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file

            if random_state is None:
                # If random_state is not provided, use the default value 
                random_state = self.random_state
        
            if split_ratio is None:
                # If split_ratio is not provided, use the default value 
                split_ratio = self.split_ratio
            

            # Get input tensors and attention masks
            input_ids_X = tokenized_X["input_ids"]
            attention_mask_X = tokenized_X["attention_mask"]
            input_ids_Y = tokenized_Y["input_ids"]
            attention_mask_Y = tokenized_Y["attention_mask"]

            # Convert NumPy arrays to torch tensors
            X_train_tensor = torch.tensor(input_ids_X.numpy(), dtype=torch.long)
            attn_mask_X_train_tensor = torch.tensor(attention_mask_X.numpy(), dtype=torch.long)
            Y_train_tensor = torch.tensor(input_ids_Y.numpy(), dtype=torch.long)

            # Split the data
            X_train, X_temp, Y_train, Y_temp, attn_mask_X_train, attn_mask_X_temp = train_test_split(
                X_train_tensor,
                Y_train_tensor,
                attn_mask_X_train_tensor,
                test_size=split_ratio,
                random_state=random_state
            )

            X_valid, X_test, Y_valid, Y_test, attn_mask_X_valid, attn_mask_X_test = train_test_split(
                X_temp,
                Y_temp,
                attn_mask_X_temp,
                test_size=0.5,
                random_state=random_state
            )

            # Convert NumPy arrays to torch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.long)
            attn_mask_X_train_tensor = torch.tensor(attn_mask_X_train, dtype=torch.long)
            Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

            X_valid_tensor = torch.tensor(X_valid, dtype=torch.long)
            attn_mask_X_valid_tensor = torch.tensor(attn_mask_X_valid, dtype=torch.long)
            Y_valid_tensor = torch.tensor(Y_valid, dtype=torch.long)

            X_test_tensor = torch.tensor(X_test, dtype=torch.long)
            attn_mask_X_test_tensor = torch.tensor(attn_mask_X_test, dtype=torch.long)
            Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)


            log(file_object=log_file, log_message=f"split the dataset into train, test and validation dataset") # logs the details about the splitting dataset into train, test and validation
            log(file_object=log_file, log_message=f"train data size: {round(len(X_train)/(len(X_train) + len(X_test) + len(X_valid)), 2)*100}%") # logs the details about the train data size
            log(file_object=log_file, log_message=f"test data size: {round(len(X_test)/(len(X_train) + len(X_test) + len(X_valid)), 2)*100}%") # logs the details about the test data size
            log(file_object=log_file, log_message=f"validation data size: {round(len(X_valid)/(len(X_train) + len(X_test) + len(X_valid)), 2)*100}%") # logs the details about the validation data size

            # Create datasets using TensorDataset
            train_dataset = TensorDataset(
                X_train_tensor, attn_mask_X_train_tensor, Y_train_tensor
            )

            validation_dataset = TensorDataset(
                X_valid_tensor, attn_mask_X_valid_tensor, Y_valid_tensor
            )

            test_dataset = TensorDataset(
                X_test_tensor, attn_mask_X_test_tensor, Y_test_tensor
            )
            
            log(file_object=log_file, log_message=f"create dictionaries for train_dataset, validation_dataset, and test_dataset") #logs the details about the dataset (train_dataset, validation_dataset, test_dataset) creation

            return train_dataset, validation_dataset, test_dataset # returns the training, validation, and test datasets # returns the training and validation and test dataset

        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            log(file_object=log_file, log_message=f"Error will be raised: {ex}") # logs the error message
            raise ex
        


    @ensure_annotations
    def remove_spaces(self, text):
        """
        Removes leading and trailing spaces from each string in the given text and capitalizes the first letter of each string.
    
        Args:
            text (str): input text.
        
        Returns:
            str: The modified strings joined with a period separator.
        
        Raises:
            Exception: If an error occurs during the process.
        """
        try:
            list_of_strings = text.split(".")
            new_list_of_strings = []
            for string in list_of_strings:
                new_string = re.sub(r'^[^a-zA-Z0-9\s]+', '', string)
                new_string = string.strip().capitalize() # remove the spaces and apply capitalization.
                new_list_of_strings.append(new_string)

            return ".".join(new_list_of_strings) # return the text after removing the spaces and applying capitalization.
    
        except Exception as ex:
            raise ex
            


        
    def add_space_before_full_stop(self, text):
        """
        Adds a space before each full stop (period) in the given text.

        Args:
            text (str): The input text.

        Returns:
            str: The modified text with spaces before full stops.

        Raises:
            Exception: If an error occurs during the process.
        """
        try:
            words = text.split(".")
            new_words = []
            for word in words:
                if word != "":
                    new_word = word + "."
                    new_words.append(new_word)

            # if new_words[-1] != ".":
            #     new_words.append(".")

            return " ".join(new_words)

        except Exception as ex:
            raise ex




if __name__ == "__main__":
    pass

