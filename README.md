# Libraries project
This is the project for the local library with a problem. Their books are being checked out and then returned late way too often. They would love to understand the cause of the issue and what they can learn from the data to proactively monitor the situation going forward.

## How to use:
Create a folder and name it "data," or set the paths in the notebooks to your folder with data .csv files.
Run the preprocess_and_visualization notebook to load the data, preprocess it, and explore it.
Train the models using the train_models notebook and check the metrics.
Investigate the data further using the feature_selection_and_shap notebook to determine the impact of each feature using AI models and SHAP analysis.

## Utils folder:
preprocess.py – Functions for preprocessing each of the .csv files and merging them into a dataset ready for model training.
visualization.py – Module for visualizing the data, with an emphasis on the ratio between books returned late and on time, in relation to independent features.
feature_selection.py – Module for sorting the features by their importance to the model.
model_training.py – Module for training AI models.
shap_stats.py – Module for implementing SHAP analysis on our data.
Required Python libraries are listed in the requirements.txt file.
