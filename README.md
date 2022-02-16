![Banner](https://github.com/italocosilva/disaster-response/blob/9295f02f7262a1bd5fa5bcf9bbb4f83bf82bce3e/app/home-page.png)

# Disaster Response: Analyzing Messages from Disasters

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Instructions to use the web app localy](#instructions)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

It was used Python 3 from Anaconda distribution.

Libraries needed:
1. numpy
2. pandas
3. pyplot
4. nltk
5. sklearn
6. lightgbm (I thing that doesn't come with Anaconda, use pip install)
7. flask

## Project Motivation<a name="motivation"></a>

For this project, I used a dataset provided by Figure Eight.

This dataset contains messages, news and social media texts from around the world in the moments next to the disaster event. 

Moreover the dataset contains 36 different categories and each message can be classified more than one of those categories. These categories help rescue teams to identify the need of the people affected by the event.

The goal of this project is to create a web app where you can input a message and discover what rescue teams may face and what resources they could need.

## File Descriptions <a name="files"></a>

1. ETL Pipeline Preparation.ipynb: Initial data exploration and preparation Jupyter Notebook.
2. ML Pipeline Preparation.ipynb: Machine Learning Pipeline Preparation, algorythm testing and hyperparameter tuning.
3. process_data.py: Organized file containing the useful discoveries from file "1.".
4. train_classifier.py: Organized file containing the winner model from file "2.".
5. run.py: back-end of the web app.

## Results<a name="results"></a>

The goal of the project was to get the best correctly predicted categories from the messages.

As the dataset has 36 different labels and one message could be more than once classified into the categories, I have to use a multi-outiput classifier.

I have tried 3 algorithms and for each one I used grid search with cross-validation to tune the hyperparameters.

The results are shown for F1 score using micro average:
1. Random Forest: 0.64
2. Linear SVC: 0.67
3. Light GBM: 0.69 (choosen one)

### Instructions to use the web app localy<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the data.
