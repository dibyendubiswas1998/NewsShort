from training import training
from predictions import prediction
from flask import Flask, request, render_template, redirect
from src.config.configuration import ConfigurationManager
import os
# from sendgrid import SendGridAPIClient
# from sendgrid.helpers.mail import Mail




app = Flask(__name__)




@app.route("/")
def Home():return render_template("index.html")



@app.route("/model_training", methods=['POST', 'GET'])
def model_training():
    cm = ConfigurationManager()
    default_model_name = cm.load_hugging_face_model_config() # get the defaul model name
    default_num_epochs = cm.model_training_config_params()[0] # get default number of epochs
    default_raw_directory_path = cm.get_raw_data_ingestion_config() # get the raw data file path
    default_data_format = "data.csv.gz" # default data format

    result = None 
    if request.method == 'POST':
        file_path = request.form['dataset_path'] 
        num_epochs = request.form['epochs_custom']
        model_name = request.form['model_name_custom']

        # if file_path is None, then use the default raw file path else provided one
        default_raw_directory_path = default_raw_directory_path if len(file_path.strip()) <1  else str(file_path).strip()
        

        # if num_epochs is None, then use the default number of epochs else provided one
        default_num_epochs = default_num_epochs if len(num_epochs.strip()) <1  else int(num_epochs.strip())
            
        #if model_name is None, then use the default model name else provided one
        default_model_name = default_model_name if len(model_name.strip()) <1 else str(model_name).strip()
            
        # start training and getting the performance matrix:
        result = training(raw_directory_path=default_raw_directory_path, model_name=default_model_name, num_train_epochs=default_num_epochs)

    dct = {
        'file_path': default_raw_directory_path,
        'data_format': default_data_format,
        'model_name': default_model_name, 
        'num_epochs': default_num_epochs,
        'score': result
    }
    print(dct)
    return render_template("model_training.html", dct = dct)



@app.route("/news_summarizations", methods=['POST', 'GET'])
def text_summarize():
    summary = None
    if request.method == 'POST':
        len_of_sentences = request.form['length']
        input_text = request.form['messages']
        
        pred_summary = prediction(text=input_text, max_length=int(len_of_sentences))
        summary = pred_summary
    return render_template("news_summarizations.html", summary=summary)


@app.route("/services")
def services():
    return render_template("services.html")


@app.route("/", methods=['POST', 'GET'])
def contact():
    if request.method == 'POST':
        full_name = request.form['full_name']
        email_id = request.form['email_id']
        phone = request.form['phone']
        messages = request.form['messages']

        
    # not provide any credentials:
    return render_template("index.html", msg="opps! unable to do that")   

    

       

    






if __name__ == '__main__':
    app.run(debug=True)

