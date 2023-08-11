from training import training
from predictions import prediction
from flask import Flask, request, render_template, redirect
from src.config.configuration import ConfigurationManager
from src.scraping.scraping import Scrapper
# from sendgrid import SendGridAPIClient
# from sendgrid.helpers.mail import Mail




app = Flask(__name__)




@app.route("/")
def Home():
    """
    Renders the home page of the web application.

    Returns:
        str: The rendered HTML template.
    """
    return render_template("index.html")



@app.route("/model_training", methods=['POST', 'GET'])
def model_training():
    """
    Handles the training of a machine learning model.

    Retrieves the default model name, number of epochs, and raw data file path from a configuration file.
    If the request method is POST, retrieves the dataset path, number of epochs, and model name from the request form.
    If these values are not provided, uses the default values.
    Calls the `training` function with the provided or default values and stores the result.
    Creates a dictionary with the file path, data format, model name, number of epochs, and result.
    Prints the dictionary.

    Returns:
        The rendered template with the dictionary as a context variable.
    """
    cm = ConfigurationManager()
    default_model_name = cm.load_hugging_face_model_config() # get the default model name
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
    """
    This function is a Flask route handler that scrapes news articles from a website, performs text summarization on the articles using a prediction model, and renders the summarized news articles on a webpage.

    :return: The rendered template with the summarized news articles.
    """
    summary = None
    # Existing code remains the same
    if request.method == 'POST':
        len_of_sentences = request.form['length']
        input_text = request.form['messages']
        
        pred_summary = prediction(text=input_text, max_length=int(len_of_sentences))
        summary = pred_summary
    return render_template("news_summarizations.html", summary=summary)



@app.route("/", methods=['POST', 'GET'])
def contact():
    """
    Handle the contact form submission.

    This function is a Flask route that is triggered when a POST request is made to the root URL ("/").
    It retrieves the values of the `full_name` and `email_id` fields from the request form.

    Returns:
        None

    Example Usage:
        @app.route("/", methods=['POST'])
        def contact():
            if request.method == 'POST':
                full_name = request.form['full_name']
                email_id = request.form['email_id']
                # process the full_name and email_id variables
    """
    if request.method == 'POST':
        full_name = request.form['full_name']
        email_id = request.form['email_id']
        phone = request.form['phone']
        messages = request.form['messages']
        
    # not provide any credentials:
    return render_template("index.html", msg="opps! unable to do that")   

    



@app.route('/domestic')
def domestic_news():
    """
    This function scrapes news articles from the NDTV website and generates a summary for each article.
    
    Returns:
        A list of dictionaries containing the title and summary of each article.
    """
    scr = Scrapper()
    news = scr.domestic_news_scrapping(url="https://www.ndtv.com/india", num_articles=5)
    new_ls = []
    for dct in news:
        dt = {
            "title": dct['title'],
            "summary": prediction(text=dct['paragraph'], max_length=150)
        }
        new_ls.append(dt)        
    return render_template("services.html", news=new_ls)




@app.route('/geopolitical')
def geopolitical_news():
    """
    This function scrapes geopolitical news articles from the NDTV website, extracts the title and summary of each article,
    and renders them on a web page using Flask and a HTML template.

    Returns:
        str: Rendered HTML template with the extracted news articles.
    """
    scr = Scrapper()
    news = scr.global_news_scrapping(url="https://www.ndtv.com/topic/geopolitical", num_articles=5)
    new_ls = []
    for dct in news:
        dt = {
            "title": dct['title'],
            "summary": prediction(text=dct['paragraph'], max_length=150)
        }
        new_ls.append(dt)    
    return render_template("services.html", news=new_ls)




if __name__ == '__main__':
    app.run(debug=True)

