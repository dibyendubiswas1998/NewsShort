from src.config.configuration import ConfigurationManager
from src.preprocessor import PreProcessor
from src.utils.logging import log 
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer



def prediction(text, max_length=150):
    """
    Generates a summary for a given text using a pre-trained summarization model.

    Args:
        text (str): The input text that needs to be summarized.
        max_length (int, optional): The maximum length of the generated summary. Defaults to 150.

    Returns:
        str: The generated summary.

    Raises:
        Exception: If any error occurs during the prediction process.
    """
    try:
        config_manager = ConfigurationManager() 
        pre_process = PreProcessor()

        training_log_file, prediction_log_file = config_manager.logs_files_config() # load prediction log file from confg.yaml file
        tokenizer_model_dir_path, summarize_model_dir_path, model_logs_dir_path = config_manager.save_model_path_config() # load tokenizer model directory path and summarize model and logs directory path

        log(file_object=prediction_log_file, log_message=f"prediction process is start") # logs the details about prediction process

        model = AutoModelForSeq2SeqLM.from_pretrained(summarize_model_dir_path) # load the model from directory
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_dir_path) # load the tokenizer

        summarization_pipeline = pipeline(
                                    task="summarization",
                                    model=model,
                                    tokenizer=tokenizer,
                                    framework="pt"
                                ) # create the pipeline
        result = summarization_pipeline(text, max_length=max_length, min_length=30, do_sample=True)
        summary_text = result[0]['summary_text']
        
        summary_text = pre_process.remove_spaces(text=summary_text) # remove extra spaces
        summary_text = pre_process.add_space_before_full_stop(text=summary_text) # add space before full stop.

        log(file_object=prediction_log_file, log_message=f"successfully predict the sumary") # logs the details about the prediction
        return summary_text # return the summary

    except Exception as ex:
        log(file_object=prediction_log_file, log_message=f"error will be: {ex}") # logs the error
        raise ex
    

    