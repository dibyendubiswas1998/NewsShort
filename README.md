# News Short: Text Summarization Web App
NewsShort is a web application designed to provide text summarization services. It utilizes advanced natural language processing techniques to generate concise and coherent summaries of input text, making it an ideal tool for quickly extracting key information from lengthy articles, documents, or any textual content.


## Features:
* **Text Summarization:** Easily summarize long blocks of text into shorter, coherent summaries while preserving essential information.

* **Model Training:** Model training for news text summarization involves creating a deep learning model using your custom data that is capable of summarizing news articles to provide concise and informative summaries.

* **User-Friendly Interface:** A clean and intuitive web interface for users to interact with the summarization tool.

* **Customization:** Adjustable parameters for summarization length, allowing users to tailor summaries according to their needs.

* **Global News Coverage:** The app keeps users informed about both local and global events. By providing news from around the world, NewsShort allows users to stay updated on international affairs and gain insights into different cultures and perspectives.


## Installation:
* **Clone from the repository:**<br>
    ```git
    $ git clone https://github.com/dibyendubiswas1998/NewsShort.git 
    ```
<br>

* **Create Conda Environment:**
    ```git
    $ conda create --name <env_name>
    ```

* **Install all the dependencies:**<br>
    ```git 
    pip install -r requirements.txt
    ```
<br>

* **Move to the project directory:**<br>
    ```git
    cd NewsShort
    ```
<br>

* **Install Git LFS:**<br>
    ```git 
    $ git lfs install
    ```
<br>

* **Fetch LFS Objects:**<br>

    ```git
    $ git lfs fetch
    ```
<br>

* **Checkout LFS Files:**<br>
    ```git
    $ git lfs checkout
    ```
<br>

* **Run the app:**<br>
    ```git
    python app.py
    ```
<br>


## Usage:
* Open your web browser and navigate to the app's URL.
* Input the text you want to summarize in the provided textarea.
* Adjust the summarization length if desired.
* Click the "Summarize" button to generate a summary.
* View the generated summary in the output section.



## Technologies Used:
* Python
* Flask
* Pytorch
* Natural Language Processing Libraries (e.g., NLTK, SpaCy)
* Hugging Face Libraries
* HTML/CSS
* JavaScript
* Git and GitHub


## Project Work Flow:
* **Step-01:** Load the raw or custom data from the particular repository, provided by user. And save the data into particular directory<br><br>

* **Step-02:** Preprocessed the raw data, like handle the missing values, duplicate values, text-preprocessing, vectorization, separate the X and Y, create tensor dataset and split them into train, test and validation sets.<br><br>

* **Step-03:** Create the model (default: t5-small), and train the model. After that save the pre-trained model & tokenizer in a particular directory.<br><br>

* **Step-04:** Evaluate the model baed on test datasets.
<br><br>

* **Step-05:** Create training and prediction pipeline for model training and prediction respectively.<br><br>

* **Step-06:** Create a Web Application for acess all the features.


<br><br><br>
If you have any questions or suggestions, feel free to reach out to me at:<br>

* https://www.linkedin.com/in/dibyendubiswas1998/
<br>

* dibyendubiswas1998 (Discord)


<br><br>

-------------------------------- Thank You
