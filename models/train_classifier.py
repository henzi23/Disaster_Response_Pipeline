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

import pickle

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Message_Categories',engine)
    X = df.message.values
    Y = df.iloc[:,4:].values
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
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
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))   
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
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
    
    print(averages)


def save_model(model, model_filepath):
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