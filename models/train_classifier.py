import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import pickle

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from lightgbm import LGBMClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    Loads the data from SQLite Database.

    Parameters
    ----------
    database_filepath : string
        Filepath of the database.

    Returns
    -------
    X : DataFrame
        Features DataFrame.
    Y : DataFrame
        Labels DataFrame.
    category_names : list
        Names of the categories.

    '''
    # creates an engine to read the data from sqlite and load in a df
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('select * from messages', engine)

    # select the 'message' col as input vars
    X = df['message']

    # select the labels
    Y = df[[col_name for col_name in df.columns if col_name not in
            ['id', 'message', 'original', 'genre']]]

    # get labels column names
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Function that covert a text to a list of tokens. It does this transformations:
        - Lower case
        - Remove ponctuation
        - Tokenize
        - Remove stopwords
        - Lemmatize

    Parameters
    ----------
    text : string
        Text to be transformed into tokens.

    Returns
    -------
    tokens : list of strings
        List of tokens.

    '''
    # put all letter to lower case
    text = text.lower()

    # substitute everything that is not letters or numbers
    text = re.sub('[^a-z 0-9]', ' ', text)

    # create tokens using nltk
    tokens = word_tokenize(text)

    # load stopwords
    stop = stopwords.words('english')

    # remove stopwords and lemmatize
    tokens = [WordNetLemmatizer().lemmatize(word)
              for word in tokens if word not in stop]

    return tokens


def build_model():
    '''
    Creates a pipeline using LGBM Classifier.

    Returns
    -------
    pipeline : Pipeline
        sk-learn Pipeline with NLP function and a multi-label classifier.

    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(LGBMClassifier()))])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Scores the model using f1_score micro average and print the result.

    Parameters
    ----------
    model : Object
        Trained model to score.
    X_test : DataFrame
        Feature Test DataFrame.
    Y_test : DataFrame
        Label Test DataFrame.
    category_names : list
        Category names.

    Returns
    -------
    None.

    '''
    # predict values for X_test
    y_pred = model.predict(X_test)

    # compute the f1_score
    score = f1_score(Y_test, y_pred, average='micro')

    print(f'f1_score micro_avg: {score}')


def save_model(model, model_filepath):
    '''
    Saves a models as a Pickle file to be used later.    

    Parameters
    ----------
    model : Object
        Trained model.
    model_filepath : string
        Filepath to save the model.

    Returns
    -------
    None.

    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
