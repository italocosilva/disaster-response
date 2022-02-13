import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # reads the data files into a pandas dataframe
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id') # merges the 2 dfs into 1
    
    return df


def clean_data(df):
    # split the categories column into many
    categories = df.categories.str.split(";", expand=True)
    
    # takes one row to find column names
    row = categories.iloc[0]
    
    # apply function to extract column names and rename the columns
    categories.columns = row.apply(lambda x: x[0:-2])
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].astype('int8')
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # limit 'related' values to 1
    df['related'] = df['related'].apply(lambda x: min(x, 1))
    
    return df

def save_data(df, database_filename):
    # create database engine
    engine = create_engine('sqlite:///' + database_filename)
    
    # save the data into the database
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()