import sys
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

def load_data(messages_filepath, categories_filepath):
    '''
    Function to load and merge message data with categorization data by ID #
    Inputs:
        messages_filepath : (string) file path to csv file with message data
        categories_filepath : (string) file path to csv file with categorization data
    Return:
        Pandas dataframe with the merged message and category information.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on = 'id')
    


def clean_data(df):
    '''
    Function to clean data by seperating out the categories column to the 36 individual measures 
    Inputs:
        df : (Pandas dataframe) dataframe containing merged message and categories data
    Return:
        Pandas dataframe with categories split out to 36 individual category columns
    '''
    
    categories = df.categories.str.split(";",expand=True)
    row = categories.iloc[[0]].squeeze()
    categories.columns = pd.Series(map(lambda i : i[:-2],row))
    
    for column in categories: 
        categories[column] = pd.to_numeric(categories[column].str[-1:])
        
    df.drop("categories",axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1) 
    return df.drop_duplicates()


def save_data(df, database_filename):
    '''
    Function to write dataframe to SQLite database
    Inputs:
        df : (Pandas dataframe) dataframe containing merged message and categories data
        database_filename : (string) filename to be given to created database
    Return:
        None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Message_Categories', engine, if_exists='replace', index=False) 


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