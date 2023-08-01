import os
import shutil
import json
import joblib
import yaml



def read_params(config_path: str) -> dict:
    """
        load the params.yaml file to extract the information.\n
        :param config_path:
        :return: params info
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
        This method helps to, if any directory is present previously then remove those directories.\n
        :param dir_path: directory path
        :return: remove directory
    """
    try:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
    except Exception as e:
        raise e


def create_dir(dirs: list):
    """
         Create the directories based on conditions.\n
        :param dirs: list_of_directories_names
        :return: create directory/ directories
    """
    try:
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
    except Exception as e:
        raise e


def save_raw_local_df(data, data_path, header=False):
    """
        save the data to specific folder.\n
        :param data: data
        :param data_path: data_path
        :param header: True or False
        :return: save data
    """
    try:
        if header:
            new_col = [col.replace(' ', "_") for col in data.columns]
            data.to_csv(data_path, index=False, header=new_col)
        else:
            data.to_csv(data_path, index=False)
    except Exception as e:
        raise e


def save_report(file_path: str, report:dict):
    """
         save the model performance report:\n
        :param file_path: file_path
        :param report: report.jso
        :return: params & score info.
    """
    try:
        with open(file_path, 'a+') as f:
            json.dump(report, f, indent=4)
    except Exception as e:
        raise e


def save_model(model_name, model_path:str):
    """
        save the model in a given directory:
        :param model: mention model
        :param model_path: model_path
        :return: save model
    """
    try:
        with open(model_path, 'wb') as f:
            joblib.dump(model_name, f)

    except Exception as e:
        raise e


if __name__ == "__main__":
    pass



