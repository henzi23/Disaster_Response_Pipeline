import sys
import pandas as pd
import re
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    '''
    Function to open connection to SQLite database, load data, and seperate into
    input variables and output categorization
    Inputs:
        database_filepath : (string) file path to SQLite database
    Return:
        X: (Pandas series) input variable (messages)
        Y: (Pandas dataframe) output categorization
        category_names: (Pandas series) output category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Message_Categories',engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
    '''
    Function take in a message and process it by standardizing, tokenizing, 
    removing stop words, and lemmatizing.
    Inputs:
        text : (string) message text
    Return:
        output_tokens : (list) resulting tokenization of the message
    '''
    text = re.sub(r"[^a-z0-9]"," ",text.lower())   
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
   
    lemmatizer = WordNetLemmatizer()
    
    output_tokens = []
    
    for token in tokens:
        output_token = lemmatizer.lemmatize(token).strip().lower()
        output_tokens.append(output_token)
        
    return output_tokens


def build_model():
     '''
    Function to build machine learning model.  Machine model includes 
    GridSearch to optimize results.
    Inputs:
        None
    Return:
        cv : model
    '''
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))   
    ])
    
    parameters = {
    'vect__min_df' : (1,5),
    'tfidf__use_idf' :  (True,False),
    'clf__estimator__n_estimators': (10,25)
    }
    
    cv = GridSearchCV(pipeline,param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to test model against test set and print model evaluation metrics
    (precision, recall, fscore) for each output category.
    Inputs:
        model: model fitted to training set
        X_test : input variables (messages) of test set
        Y_test : output categorizations for test set
        category_names: (Pandas series) output category names
    Return:
        None, but function prints out evaluation metrics
    '''
    y_pred = model.predict(X_test)
    
    y_pred_df = pd.DataFrame(y_pred)
    y_test_df = pd.DataFrame(Y_test)
    
    results = []

    for cat in range(len(y_pred[0])):
        precision,recall,fscore,support = score(y_test_df[cat],y_pred_df[cat],average='weighted') 
        results.append((category_names[cat],precision,recall,fscore))
        
    results = pd.DataFrame(results,columns=('Category','Precision','Recall','fscore'))
    averages = pd.DataFrame([['Categories Average', results['Precision'].mean(),
           results['Recall'].mean(), results['fscore'].mean()]], columns = results.columns)
    
    print(results.append(averages, ignore_index=True))


def save_model(model, model_filepath):
    '''
    Function to save model as a pickle file
    Inputs:
        model_filepath : (string) file path to save model
    Return:
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()