import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

# Project Structure:
project_name = "NewsShort"
list_of_files = [
    "main.py", # main.py file
    "setup.py", # setup.py file
    "app.py", # app.py file
    "requirements.txt", # requirements.txt file
    "notebook/.gitkeep", # notebook directory
    "artifacts/.gitkeep", # artifacts directory
    "artifacts/preprocessed_data/.gitkeep", # artifacts/preprocessed_data directory
    "artifacts/model/.gitkeep", # artifacts/model directory
    "artifacts/report/performance_report.json", # artifacts/report directory for performance report
    "logs/trainings_logs.txt", # logs/trainings_logs.txt file
    "logs/predictions_logs.txt", # logs/predictions_logs.txt file
    "params.yaml", # params.yaml file
    "raw_data/.gitkeep", # raw_data directory
    "prediction_file/.gitkeep", # prediction_file

    "src/__init__.py", # src package
    "src/utils/__init__.py", # utils package

    "src/utils/common_utils.py", # common_utils.py file
    "src/utils/logging.py", # logging.py file

    "src/load_and_save_data.py", # load_and_save_data.py file
    "src/data_augmentation.py", # data_augmentation.py file
    "src/preprocessor.py", # preprocessor.py file
    "src/model_creation.py", # model_creation.py file
    "src/training.py", # training.py file
    "src/predictions.py", # predictions.py file
    "src/evaluation.py", # evaluation.py file
    # templates Website
    "templates/index.html",
    "static/css/index.css",
    "static/js/index.js"
]


def create_project_template(project_template_lst):
    try:
        for filepath in project_template_lst:
            filepath = Path(filepath)
            file_dir, file_name = filepath.parent, filepath.name

            if file_dir != "":
                Path(file_dir).mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {file_dir}")

            if (not filepath.exists()) or (filepath.stat().st_size == 0):
                filepath.touch()
                logging.info(f"Created file: {filepath}")
            else:
                logging.info(f"{file_name} already exists")

    except (OSError, IOError) as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.error(f"Unknown error: {e}")
        



if __name__ == "__main__":
    logging.info(f"Created project template for: {project_name}")
    create_project_template(list_of_files)
