from src.utils.logging import log
from src.utils.common_utils import read_params
import os



    
class ConfigurationManager:
    def __init__(self, config_yaml_file="params.yaml"):
        self.config = read_params(config_path=config_yaml_file)
    

    def logs_files_config(self):
        """
        Retrieve the paths of the training and prediction log files from the configuration file.

        Returns:
            tuple: A tuple containing the path of the training log file and the path of the prediction log file.

        Raises:
            Exception: If the retrieval of the log file paths fails.
        """
        try:
            self.training_log_file = self.config['logs_files']['trainings_logs'] # training logs file
            self.prediction_log_file = self.config['logs_files']['predictions_logs'] # prediction logs file
            return self.training_log_file, self.prediction_log_file
        except Exception as ex:
            raise ex
        

    def get_raw_data_ingestion_config(self):
        """
        Retrieve the path of the raw data directory from the configuration file.

        Returns:
            str: The path of the raw data directory.
        """
        try:
            self.get_raw_data_file_path = self.config['raw_data_source']['raw_data_dir']
            return self.get_raw_data_file_path
        except Exception as ex:
            raise ex


    def saving_data_ingestion_config(self):
        """
        Retrieve the root directory and file name for saving the raw data from the configuration file.

        Returns:
        Tuple: The root directory and file name for saving the raw data.

        Raises:
        Exception: If there is an error retrieving the values from the configuration file.
        """
        try:
            self.new_raw_data_dir = self.config['data_saveing']['root_dir']
            self.new_raw_data_file_name = self.config['data_saveing']['file_name']
            return self.new_raw_data_dir, self.new_raw_data_file_name
        except Exception as ex:
            raise ex


    def get_raw_data_from_artifacts(self):
        """
        Retrieves the path of the raw data file from the configuration file.

        Returns:
            str: The path of the raw data file.

        Raises:
            Exception: If there is an error retrieving the path from the configuration file.
        """
        try:
            self.raw_data_path = self.config['data_saveing']['file_name']
            return self.raw_data_path
        except Exception as ex:
            raise ex


    def hugging_face_dataset_config(self):
        """
        Retrieve the name of the file containing the dataset from the configuration file.

        Returns:
            str: The name of the file containing the dataset.

        Raises:
            Exception: If the dataset file name cannot be retrieved from the configuration file.
        """
        try:
            self.huggingface_dataset = self.config['load_data_from_hugging_face']['file_name'] # data set_name
            return self.huggingface_dataset
        except Exception as ex:
            raise ex
    

    def load_hugging_face_model_config(self):
        """
        Retrieves the default and custom model names for transfer learning and fine-tuning from the configuration file.

        Returns:
            tuple: A tuple containing the default model name for transfer learning and the custom model name for fine-tuning.

        Raises:
            Exception: If there is an error retrieving the model names from the configuration file.
        """
        try:
            self.default_model = self.config['model_definition']['default']['model_name'] # default model for fine-tuning
            return self.default_model  # return default model for fine-tuning
        except Exception as ex:
            raise ex
    


    def get_output_col_name_config(self) -> str:
        """
        Retrieves the name of the output column from the configuration file.

        Returns:
            str: The name of the output column.

        Raises:
            Exception: If the output column name is not found in the configuration file.
        """
        try:
            self.output_col = self.config['data_definition']['output_col'] # get the output_col name
            self.input_col = self.config['data_definition']['input_col'] # get the input_col name
            return self.output_col, self.input_col # return the output_col name and the input_col name
        except Exception as ex:
            raise ex



    def random_split_config(self):
        """
        Retrieves the split ratio and random state values from the configuration file.

        Returns:
            tuple: A tuple containing the split ratio and random state values.

        Raises:
            Exception: If there is an error retrieving the split ratio and random state values from the configuration file.
        """
        try:
            self.splt_ratio = self.config['data_split']['split_ratio'] # split ratio
            self.random_state = self.config['data_split']['random_state'] # random state
            return self.splt_ratio, self.random_state # return the splt ratio and the random state
        except Exception as ex:
            raise ex
        

    def save_model_path_config(self):
        """
        Retrieves the directory paths for saving the tokenizer and summarized models from the configuration file.

        Returns:
        A tuple containing the directory path for saving the tokenizer model and the directory path for saving the summarized model.
        """
        try:
            self.tokenizer_model_dir_path = self.config['model_svaing']['tokenizer_model']['root_dir'] # tokenizer_model directory
            self.summarize_model_dir_path = self.config['model_svaing']['summarized_model']['root_dir'] # summation_model directory
            self.model_logs_dir_path = self.config['model_svaing']['summarized_model']['logs_dir'] # summation_model logs directory
            return self.tokenizer_model_dir_path, self.summarize_model_dir_path, self.model_logs_dir_path # return tokenizer_model_dir_path and summation_model_dir_path and model_logs_dir_path
        except Exception as ex:
            raise ex



    def model_training_config_params(self):
        """
        Retrieve the hyperparameters required for model training.

        This method retrieves the hyperparameters, such as the number of epochs, batch size, weight decay, and evaluation strategy,

        Returns:
            A tuple containing the following hyperparameters:
            - Number of training epochs
            - Warmup steps
            - Batch size for training
            - Batch size for evaluation
            - Weight decay
            - Logging steps
            - Evaluation strategy
            - Evaluation steps
            - Save steps
            - Gradient accumulation steps

        Raises:
            Exception: If there is an error while retrieving the hyperparameters from the configuration file.
        """
        try:
            self.num_train_epochs = self.config['TrainingArguments']['num_train_epochs'] # number_of_epochs
            self.warmup_steps = self.config['TrainingArguments']['warmup_steps'] # warmup_steps
            self.per_device_train_batch_size = self.config['TrainingArguments']['per_device_train_batch_size'] # per_device_train_batch_size
            self.per_device_eval_batch_size = self.config['TrainingArguments']['per_device_eval_batch_size'] # per_device_eval_batch_size
            self.weight_decay = self.config['TrainingArguments']['weight_decay'] # weight_decay
            self.logging_steps = self.config['TrainingArguments']['logging_steps'] # logging_steps
            self.evaluation_strategy = self.config['TrainingArguments']['evaluation_strategy'] # evaluation_strategy
            self.eval_steps = self.config['TrainingArguments']['eval_steps'] # evaluation_steps
            self.save_steps = self.config['TrainingArguments']['save_steps'] # save_steps
            self.gradient_accumulation_steps = self.config['TrainingArguments']['gradient_accumulation_steps'] # gradient_accumulation_steps

            return self.num_train_epochs, self.warmup_steps, self.per_device_train_batch_size, self.per_device_eval_batch_size, self.weight_decay, self.logging_steps, self.evaluation_strategy, self.eval_steps, self.save_steps, self.gradient_accumulation_steps # retun all the hyperparameters
        except Exception as ex:
            raise ex
        


    def read_performance_report_config(self):
        """
        Retrieves the directory path and file name for saving the performance report from the configuration file.

        Returns:
            tuple: A tuple containing the directory path and file name for saving the performance report.

        Raises:
            Exception: If there is an error during the retrieval process.

        Example Usage:
            config_manager = ConfigurationManager()
            report_dir, report_file = config_manager.read_predormace_report_config()
            print(report_dir)  # Output: The directory path for saving the performance report
            print(report_file)  # Output: The file name for saving the performance report
        """
        try:
            self.report_dir = self.config['performace_report']['root_dir'] # report directory
            self.report_file = self.config['performace_report']['report_file'] # performace_report.json file
            return self.report_dir, self.report_file # return directory and file name
        except Exception as ex:
            raise ex
        
    