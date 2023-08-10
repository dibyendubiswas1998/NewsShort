from src.utils.logging import log
from src.config.configuration import ConfigurationManager
from src.load_and_save_data import DataLoader
from src.preprocessor import PreProcessor
from src.model_creation_and_training import ModelCreationTraining
from src.evaluation import ModelEvaluation
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer





def training(raw_directory_path=None, model_name=None, num_train_epochs=None, warmup_steps=None, per_device_train_batch_size=None, 
             per_device_eval_batch_size=None, weight_decay=None, logging_steps=None, 
             evaluation_strategy=None, eval_steps=None, save_steps=None, gradient_accumulation_steps=None):
    """
    Perform training, evaluation, and logging steps for a machine learning model.

    Args:
        raw_directory_path (str): Path to the directory of raw data files
        model_name (str): The name of the model.
        num_train_epochs (int): The number of training epochs.
        warmup_steps (int): The number of warmup steps.
        per_device_train_batch_size (int): The batch size for training.
        per_device_eval_batch_size (int): The batch size for evaluation.
        weight_decay (float): The weight decay value.
        logging_steps (int): The number of steps between logging.
        evaluation_strategy (str): The evaluation strategy.
        eval_steps (int): The number of steps between evaluation.
        save_steps (int): The number of steps between saving.
        gradient_accumulation_steps (int): The number of steps for gradient accumulation.

    Returns:
        dict: The average Rouge scores.

    Raises:
        Exception: If an error occurs during the training process.
    """
    try:
        config_manager = ConfigurationManager() 
        training_log_file, prediction_log_file = config_manager.logs_files_config() # load training log file from confg.yaml file
        tokenizer_model_dir_path, summarize_model_dir_path, model_logs_dir_path = config_manager.save_model_path_config() # load tokenizer model directory path and summarize model and logs directory path

        """Step 1: Load and Save the dataset"""
        data_loader = DataLoader()
        log(file_object=training_log_file, log_message=f"Step 1: load and save process is start") # logs the details about Step 1

        # load the dataset
        data = data_loader.load_csv_data(directory_path=raw_directory_path, log_file=None) 

        # save dataset in a particular directory.
        data_loader.save_data(data=data, new_raw_data_dir=None, new_data_file_name=None, log_file=None) 
        log(file_object=training_log_file, log_message=f"Step 1: Load and Save datset process is successfully completed\n") # logs the details of the process completation 


        """Step 2: Pre-processed the data"""
        pre_process = PreProcessor()
        log(file_object=training_log_file, log_message=f"Step 2: Pre-processing step is start") # logs the details about Step 2
        
        # handle the data (handle the missing data)
        data = pre_process.handle_data(data=None, log_file=None) 
        
        # separate the input and output as list of strings.
        X, Y = pre_process.separate_x_y(data=data, input_col_name=None, output_col_name=None, log_file=None) 
        
        # perform the text processing on the input data and output data
        # X = pre_process.text_preprocessing(data=X, log_file=None) 
        # Y = pre_process.text_preprocessing(data=Y, log_file=None) 
        
        # get the neumeric representation of the the input and output data
        tokenized_X, tokenized_Y = pre_process.word_embedding(X=list(X), Y=list(Y), log_file=None, model_name=None, tokenizer_model_dir_path=None) 
        
        # # split the data into tran, test and validation tensors datasets.
        train_dataset, validation_dataset, test_dataset = pre_process.split_and_create_tensor_(tokenized_X=tokenized_X, tokenized_Y=tokenized_Y, random_state=None, split_ratio=None, log_file=None)
        log(file_object=training_log_file, log_message=f"Step 2: Pre-processing steps are successfully completed\n") # logs the details of the process completation 
       

        # """Step 3: Model Creation and Training (Fine-Tuning approach)"""
        model_training = ModelCreationTraining()
        log(file_object=training_log_file, log_message=f"Step 3: Model Creation and Training (Fine-Tuning approach) process is start") # logs the details about Step 3

        # # start model training process
        model_training.model_training(training_dataset=train_dataset, 
                                      validation_dataset=validation_dataset, 
                                      num_train_epochs=num_train_epochs, 
                                      warmup_steps=warmup_steps, 
                                      per_device_train_batch_size=per_device_train_batch_size, 
                                      per_device_eval_batch_size=per_device_eval_batch_size,
                                      weight_decay=weight_decay, 
                                      logging_steps=logging_steps, 
                                      evaluation_strategy=evaluation_strategy, 
                                      eval_steps=eval_steps, 
                                      save_steps=save_steps, gradient_accumulation_steps=gradient_accumulation_steps,
                                      model_name=model_name, 
                                      tokenizer_model_path=None, 
                                      summarizer_model_path=None, 
                                      model_logs_path=None, 
                                      log_file=None)

        log(file_object=training_log_file, log_message=f"Step 3: Model Creation and Training (Fine-Tuning approach) process is successfully completed\n") # logs the details of the process completation 


        # """Step 4: Model Evaluation"""
        evaluation = ModelEvaluation()
        log(file_object=training_log_file, log_message=f"Step 4: Model Evaluation process is start") # logs the details about Step 4

        model = AutoModelForSeq2SeqLM.from_pretrained(summarize_model_dir_path) # load the model from directory
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_dir_path) # load the tokenizer

        test_list_of_scores = evaluation.list_of_get_performance_score(dataset=test_dataset, model=model, tokenizer=tokenizer, log_file=None) # get the list of dcores

        average_rouge_scores = evaluation.get_average_score_and_save(list_of_score=test_list_of_scores, report_file_path=None, log_file=None) # get the average score

        log(file_object=training_log_file, log_message=f"Step 4: Model Evaluation process is successfully completed\n") # logs the details of the process completation 

        return average_rouge_scores # return the average scores

    except Exception as ex:
        log(file_object=training_log_file, log_message=f"error will be: {ex}") # logs the error
        raise ex
    
