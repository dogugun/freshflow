# Freshflow Interview Case
This document is to explain the simple project structure. 

Project is formed of 2 parts, one notebook and 2 scripts. 

Notebook is developed for the sake of data analysis.

- train.py: the script to build the model and train. The resulting GBM model is saved to the same directory.
- prediction.py: makes a scoring for the next day with the given inpu data, and the result is printed on the console
