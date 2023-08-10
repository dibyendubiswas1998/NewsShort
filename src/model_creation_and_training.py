import os
from src.utils.logging import log
from src.utils.common_utils import clean_prev_dirs_if_exis, create_dir
from src.config.configuration import ConfigurationManager
from ensure import ensure_annotations
import torch
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments



class ModelCreationTraining:
    """
        This class is responsible for creating and training a sequence-to-sequence language model for fine-tuning. It loads the configuration settings from the ConfigurationManager class and uses them to create and train the model. The class provides methods for creating the model, custom data collation, and model training.

        Methods:
            - create_model(): Creates a sequence-to-sequence language model for fine-tuning. It loads the  pre-trained model specified by the 'model_name' parameter or uses the default model for fine-tuning if the parameter is not provided. The method logs the details about the model loading and returns the loaded pre-trained model.

            - data_collator(): A custom data collator that loads the data sequence-wise. It takes a list of features, where each feature is a tuple of input_ids, attention_mask, and labels, and returns a dictionary containing the stacked input_ids, attention_mask, and labels with keys 'input_ids', 'attention_mask', and 'labels', respectively.
            
            - model_training(): Trains the sequence-to-sequence language model. It takes the training and validation datasets, the number of training epochs, the batch sizes, the weight decay, the logging and evaluation steps, the evaluation strategy, the save steps, the gradient accumulation steps, and the paths for the tokenizer model, the summarizer model, and the model logs. The method creates the model, loads the tokenizer, and trains the model using the Trainer class from the transformers library. It logs the details about the model training and saves the trained model and the tokenizer model.

        Fields:
            - config_manager: An instance of the ConfigurationManager class that loads the configuration settings.
            - training_log_file: The path to the training log file.
            - prediction_log_file: The path to the prediction log file.
            - default_model_ft: The name of the default pre-trained model for fine-tuning.
            - tokenizer_model_dir_path: The path to the directory where the tokenizer model will be saved.
            - summarize_model_dir_path: The path to the directory where the trained model will be saved.
            - model_logs_dir_path: The path to the directory where the model logs will be saved.
        """
    def __init__(self):
        self.config_manager = ConfigurationManager()    # load ConfigurationManager class
        self.training_log_file, self.prediction_log_file = self.config_manager.logs_files_config() # load training log file from confg.yaml file
        self.default_model_ft = self.config_manager.load_hugging_face_model_config() #  load default model for fine-tuning
        self.tokenizer_model_dir_path, self.summarize_model_dir_path, self.model_logs_dir_path = self.config_manager.save_model_path_config() # load tokenizer model directory path and summarize model and logs directory path
        self.num_train_epochs, self.warmup_steps, self.per_device_train_batch_size, self.per_device_eval_batch_size, self.weight_decay, self.logging_steps, self.evaluation_strategy, self.eval_steps, self.save_steps, self.gradient_accumulation_steps = self.config_manager.model_training_config_params() # returns all the hyperparameters

    

    @ensure_annotations
    def create_model(self, model_name=None, log_file=None):
        """
        Create a sequence-to-sequence language model for fine-tuning.

        Args:
            model_name (str, optional): The name of the pre-trained model to be loaded. If not provided, the default model for fine-tuning is used.
            log_file (file object, optional): The log file to be used for logging. If not provided, the default log file from the ConfigurationManager class is used.

        Returns:
            model (PyTorch model object): The loaded pre-trained model.

        Raises:
            Exception: If an error occurs during the model loading process.

        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
        
            if model_name is None:
                # If model_name is not provided then use the default model for fine-tuning
                model_name = self.default_model_ft
        
            device = "cuda" if torch.cuda.is_available() else "cpu" # check which one available: cpu or cuda (gpu)

            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device) # logd the model
            log(file_object=log_file, log_message=f"load the model, model name: {model_name}") # logs the details about model loadding

            return model # return the model

        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            log(file_object=log_file, log_message=f"Error will be raised: {ex}") # logs the error message
            raise ex




    @ensure_annotations
    def data_collator(self, features):
        """
        Custom data collator that loads the data sequence-wise.

        Args:
            features (List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): A list of features, where each feature is a tuple of input_ids, attention_mask, and labels.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the stacked input_ids, attention_mask, and labels with keys 'input_ids', 'attention_mask', and 'labels', respectively.

        Raises:
            Exception: If an error occurs during data collation.

        """
        try:
            input_ids = torch.stack([feature[0] for feature in features])
            attention_mask = torch.stack([feature[1] for feature in features])
            labels = torch.stack([feature[2] for feature in features])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        except Exception as ex:
            raise ex
    

    


    @ensure_annotations
    def model_training(self, training_dataset, validation_dataset, num_train_epochs=None,
                       warmup_steps=None, per_device_train_batch_size=None, per_device_eval_batch_size=None,
                       weight_decay=None, logging_steps=None, evaluation_strategy=None, eval_steps=None, save_steps=None, gradient_accumulation_steps=None,
                       model_name=None, tokenizer_model_path=None, summarizer_model_path=None, model_logs_path=None, log_file=None):
        """
            Trains a sequence-to-sequence language model for fine-tuning using the Hugging Face Transformers library.

            Args:
                training_dataset (Dataset): The dataset used for training the model.
                validation_dataset (Dataset): The dataset used for validating the model.
                num_train_epochs (int): The number of epochs for which the model will be trained.
                warmup_steps (int): The number of warmup steps for the optimizer.
                per_device_train_batch_size (int): The batch size for training.
                per_device_eval_batch_size (int): The batch size for evaluation.
                weight_decay (float): The weight decay for the optimizer.
                logging_steps (int): The number of steps after which the logs will be printed.
                evaluation_strategy (str): The evaluation strategy used during training.
                eval_steps (int): The number of steps after which the evaluation will be performed.
                save_steps (int): The number of steps after which the model will be saved.
                gradient_accumulation_steps (int): The number of gradient accumulation steps.
                model_name (str, optional): The name of the pre-trained model to be loaded. Defaults to None.
                tokenizer_model_path (str, optional): The path where the tokenizer model will be saved. Defaults to None.
                summarizer_model_path (str, optional): The path where the trained model will be saved. Defaults to None.
                model_logs_path (str, optional): The path where the logs will be saved. Defaults to None.
                log_file (str, optional): The log file to be used for logging. Defaults to None.

            Returns:
                trained_model: The trained model.
                tokenizer: The saved tokenizer.
                trained_model: The saved trained model.
        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
        
            if model_name is None:
                # If model_name is not provided then use the default model for fine-tuning
                model_name = self.default_model_ft
        
            if tokenizer_model_path is None:
                # if tokenizer_model_path is not provided, use the default tokenizer_model_path
                tokenizer_model_path = self.tokenizer_model_dir_path
            # clean_prev_dirs_if_exis(dir_path=tokenizer_model_path) # remove the old directory if it exists
            # create_dir(dirs=[tokenizer_model_path]) # create the directory new one
        
            tokenizer = AutoTokenizer.from_pretrained(model_name) # load the tokenizer 

            if summarizer_model_path is None:
                # if summarizer_model_path is not provided, use the default path
                summarizer_model_path = self.summarize_model_dir_path
            # clean_prev_dirs_if_exis(dir_path=summarizer_model_path) # remove directory if it exists
            # create_dir(dirs=[summarizer_model_path]) # create directory new one

            if model_logs_path is None:
                # if model_logs_path is not provided, use the default path
                model_logs_path = self.model_logs_dir_path
            # clean_prev_dirs_if_exis(dir_path=model_logs_path) # remove directory if it exists
            # create_dir(dirs=[model_logs_path]) # create directory new one
            
                 
            model = self.create_model(model_name=model_name) # create the model

            # all the hyperparameters:warmup_stepswarmup_warmup_steps
            if num_train_epochs is None: num_train_epochs = self.num_train_epochs
            if warmup_steps is None: warmup_steps = self.warmup_steps
            if per_device_train_batch_size is None: per_device_train_batch_size = self.per_device_train_batch_size
            if per_device_eval_batch_size is None: per_device_eval_batch_size = self.per_device_eval_batch_size
            if weight_decay is None: weight_decay = self.weight_decay
            if logging_steps is None: logging_steps = self.logging_steps
            if evaluation_strategy is None: evaluation_strategy = self.evaluation_strategy
            if eval_steps is None: eval_steps = self.eval_steps
            if save_steps is None: save_steps = self.save_steps
            if gradient_accumulation_steps is None: gradient_accumulation_steps = self.gradient_accumulation_steps


            trainer_args = TrainingArguments(
                                output_dir=model_logs_path, 
                                num_train_epochs=num_train_epochs,
                                warmup_steps=warmup_steps,
                                per_device_train_batch_size=per_device_train_batch_size, 
                                per_device_eval_batch_size=per_device_eval_batch_size,
                                weight_decay=weight_decay, 
                                logging_steps=logging_steps,
                                evaluation_strategy=evaluation_strategy, 
                                eval_steps=eval_steps, 
                                save_steps=save_steps,
                                gradient_accumulation_steps=gradient_accumulation_steps
                            )


            trainer = Trainer(
                            model=model, 
                            args=trainer_args,
                            tokenizer=tokenizer, 
                            data_collator=self.data_collator,
                            train_dataset=training_dataset,
                            eval_dataset=validation_dataset
                        )

            trainer.train() # train the model
            log(file_object=log_file, log_message=f"successfully trained model") # logs the details about the model trainings

            tokenizer.save_pretrained(tokenizer_model_path) # saves the tokenizer model
            log(file_object=log_file, log_message=f"save the tokenizer into {tokenizer_model_path}") # logs the details about the tokenizer model saving

            model.save_pretrained(summarizer_model_path) # saves the trained model
            log(file_object=log_file, log_message=f"save the trained model into {summarizer_model_path}") # logs the details about the trained model saving
        

        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            log(file_object=log_file, log_message=f"Error will be raised: {ex}") # logs the error message
            raise ex



if __name__ == "__main__":
    pass
    
