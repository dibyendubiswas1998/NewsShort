from src.utils.logging import log 
from src.utils.common_utils import save_report
from src.config.configuration import ConfigurationManager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rouge_score import rouge_scorer
from ensure import ensure_annotations
import numpy as np
import torch



class ModelEvaluation:
    def __init__(self):
        self.config_manager = ConfigurationManager()    # load ConfigurationManager class
        self.training_log_file, self.prediction_log_file = self.config_manager.logs_files_config() # load training log file from confg.yaml file
        self.report_dir, self.report_file_path = self.config_manager.read_performance_report_config() # load report directory and report file
        self.tokenizer_model_dir_path, self.summarize_model_dir_path, self.model_logs_dir_path = self.config_manager.save_model_path_config() # load tokenizer model directory path and summarize model and logs directory path



    
    @ensure_annotations
    def list_of_get_performance_score(self, dataset, model=None, tokenizer=None, log_file=None):
        """
        Calculate the performance scores for a given dataset using a specified model and tokenizer.
    
        Args:
            dataset (list): A list of data instances.
            model (optional): The model to use for generating summaries. If not provided, a pre-trained model will be loaded.
            tokenizer (optional): The tokenizer to use for decoding summaries. If not provided, a tokenizer will be loaded from a directory.
            log_file (optional): The log file to write log messages to. If not provided, a default log file will be used.
    
        Returns:
            list: A list of performance scores for each data instance in the dataset.
    
        Raises:
            Exception: If an error occurs during the execution of the method.
        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
        
            if tokenizer is not None:
                # If tokenizer is not provided, then load the tokenizer from the directory
                tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_dir_path) # load the tokenizer
        
            if model is None:
                # If model is not provided, load the pretrained model from the dictionary
                model = AutoModelForSeq2SeqLM.from_pretrained(self.summarize_model_dir_path) # load the model from the dictionary
                
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True) # define the default scocre
            list_of_score = [] # store the scores one-by-one

            for i in range(len(dataset)):
                input_input_ids = dataset[i][0]
                input_attention_mask = dataset[i][1]
                target_input_ids = dataset[i][2]
            
                generated_ids = model.generate(input_ids=input_input_ids.unsqueeze(0).to(model.device),
                                                        attention_mask=input_attention_mask.unsqueeze(0).to(model.device),
                                                        max_length=200,  
                                                        num_beams=4,   
                                                        early_stopping=True)

                pred_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

                target_summary = tokenizer.decode(target_input_ids, skip_special_tokens=True)
                score = scorer.score(target_summary, pred_summary)
                list_of_score.append(score)
                
        
            log(file_object=log_file, log_message=f"return the validation score") # logs the details about get list of score
            return list_of_score # return all the scores

    
        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            log(file_object=log_file, log_message=f"Error will be raised: {ex}") # logs the error message
            raise ex


    


    @ensure_annotations
    def get_average_score_and_save(self, list_of_score, report_file_path=None, log_file=None):
        """
        Calculates the average precision, recall, and F-measure scores for a given list of scores.

        Args:
            list_of_score (list): A list of scores.
            log_file (str, optional): The path to the log file. If not provided, the default value from ConfigurationManager() will be used.

        Returns:
            dict: A dictionary containing the average precision, recall, and F-measure scores for each metric.

        Raises:
            Exception: If an error occurs during the execution of the method.

        """
        try:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            
            if report_file_path is None:
                # if report_file_path is not provided, use the default file path
                report_file_path = self.report_file_path
         
            average_rouge_scores = {} # store all the score as dictionary format

            for metric in ["rouge1", "rouge2", "rougeL", "rougeLsum"]: # for each metric get the score
                precision_list = [] # precision list
                recall_list = [] # recall list
                fmeasure_list = [] # fmeasure list

                for rouge_score in list_of_score: # iterate score one-by-one
                    precision_list.append(rouge_score[metric][0])
                    recall_list.append(rouge_score[metric][1])
                    fmeasure_list.append(rouge_score[metric][2])
                
                average_precision = round(np.mean(precision_list), 4) # calculate average_precision
                average_recall = round(np.mean(recall_list), 4) # calculate average_recall
                average_fmeasure = round(np.mean(fmeasure_list), 4) # calculate average_fmeasure

                average_rouge_scores[metric] = {
                    "precision": average_precision,
                    "recall": average_recall,
                    "fmeasure": average_fmeasure
                } # store the metric as dictionary format

            save_report(file_path=report_file_path, report=average_rouge_scores) # save the report
            log(file_object=log_file, log_message=f"save the report to {report_file_path}") # logs the details about saving the report
            return average_rouge_scores # return the average score

        except Exception as ex:
            if log_file is None:
                # If log_file is not provided, use the default value from ConfigurationManager()
                log_file = self.training_log_file
            log(file_object=log_file, log_message=f"Error will be raised: {ex}") # logs the error message
            raise ex
        
