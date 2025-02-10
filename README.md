# libraries

This is the project for thr local library with a problem. Their books are being checked out and then returned late way too often. They would love to understand the cause of the issue and what they can learn from the data to proactively monitor the situation going forward.

## How to use:

- Create folder and name it "data" or set the paths in the notebooks to your folder with data .csv files
- Run the preprocess_and_visualization notebook to load the data, preprocess it and explore it
- Train the models using train_models notebook end check the metrics
- Investigate the data further using feature_selection_and_shap notebook to determine the impact of each feature using AI models and Shap analysis

## Utils folder:
  - preprocess.py - functions for preprocessing each of the csv files, and merging them in dataset ready for model training
  - visualization.py - module for visualizing the data with accent on the ratio between book returned late and returned on time, in relation to independent features
  - feature_selection.py - module for sorting the features by their importance to model
  - model_training.pt - module for training AI models
  - shap_stats.py - module for implementing Shap analysis on our data
    
  Required Python libraries are in the requirements.txt file
  
