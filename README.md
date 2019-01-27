# Disaster Response Pipeline

Project to create a pipelines to process and analyze disaster message data (data provided by Figure Eight).  The outputs from these pipelines are used to provide a Flask web app visualization.  The components of the project are:
* ETL pipeline - load datasets (flat files), transform them and store in SQLite database
* ML Pipeline - loads data from SQLite database, fits/tunes a text processing machine learning model, and exports the model
* Web app - creates a web app to show results

## Getting Started

This repo includes 4 main folders:
* Preparation - this contains Jupyter notebooks detailing the steps in developing the ETL and ML Pipelines.  This is included as reference only.
* data - this contains the Python script for the ETL pipeline.  There are also sample data flat files included for reference.
* model - this contains the Python script for the ML pipeline.
* app - This contains the files to generate the Flask web app to share visualizations regarding the data and predictions.

### Prerequisites

All programming utilizes Python or HTML.  The Anconda environment for the python scripting is provided in file environment.yml for reference.

### Installing

To run the project files copy folders (and associated content) app, data, and models to a single directory on your local machine.  Be sure not to change any file names.  From local command line (Anaconda environment suggested) navigate to the directory holding these folders.


## Deployment

In the command line (in the directory housing the folders installed):
* Run ETL pipeline with command `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
* Run ML piepline with command `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
* Navigate to folder app and run command 'python run.py' to start web app
* To view web app go into web browser and type 'localhost:3001'


## Author

* James Henzi


## Acknowledgments

* Udacity - for project inspiration and guidance
* Figure Eight - for project inspiration and cleaned source data files
