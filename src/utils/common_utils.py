import os
import shutil
import json
import yaml



    
def read_params(config_path: str) -> dict:
    """
    Read the YAML file at the specified path and return its contents as a dictionary.

    Args:
        config_path (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.

    Raises:
        Exception: If there is an error while reading the file.

    """
    try:
        with open(config_path) as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except Exception as e:
        print(e)
        raise e


    
def clean_prev_dirs_if_exis(dir_path: str):
    """
    Removes a directory and all its contents if it exists.

    This function takes a directory path as input and checks if the directory exists. If it does, it removes the directory and all its contents using the `shutil.rmtree` function.

    :param dir_path: The path of the directory to be checked and removed.
    :type dir_path: str
    :raises: Exception: If any exception occurs during the removal process.
    """
    try:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
    except Exception as e:
        raise e


    

def create_dir(dirs: list):
    """
    Create directories based on the provided list of directory names.
    
    This function takes a list of directory names as input and creates those directories if they don't already exist.
    
    :param dirs: A list of directory names to be created.
    :type dirs: list
    :raises Exception: If any exception occurs during the creation of directories.
    """
    try:
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
    except Exception as e:
        raise e

    

def save_raw_local_df(data, data_path, header=False):
    """
    Save a pandas DataFrame to a specific file path.

    Parameters:
    data (DataFrame): The DataFrame to be saved.
    data_path (str): The file path where the DataFrame will be saved.
    header (bool, optional): Whether to include column names as the first row in the saved file. Defaults to False.

    Raises:
    Exception: If any error occurs during the saving process.

    Returns:
    None
    """
    try:
        if header:
            new_col = [col.replace(' ', "_") for col in data.columns]
            data.to_csv(data_path, index=False, header=new_col)
        else:
            data.to_csv(data_path, index=False)
    except Exception as e:
        raise e


    

def save_report(file_path: str, report: dict):
    """
    Save the model performance report to a file in JSON format.

    This function takes a file path and a dictionary as input. It opens the file specified by the file path and appends the dictionary as a JSON object to the file. If an error occurs during the process, it raises an exception.

    :param file_path: The path to the file where the report will be saved.
    :param report: A dictionary containing the model performance information.
    :return: None
    :raises: Exception if an error occurs during the file saving process.
    """
    try:
        with open(file_path, 'a+') as f:
            json.dump(report, f, indent=4)
    except Exception as e:
        raise e



if __name__ == "__main__":
    pass



