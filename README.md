# Final Project - Martin Nelson

## How to run
1. cd into "Nelson_Martin_Final_Project" directory
2. run pip install requirements.txt, or ensure your environment has the requirements downloaded (optional)
3. cd into the "Project" directory
4. execute python explore.py to generate some plots already included in the plots folder
5. execute python analysis.py to run the models and generate plots in the rest of the plots folders

## File Descriptions

### 1. randomize.py
- This file takes any tracks listed in the tracks.txt file and randomizes the order, outputting into the tracks_output.txt - will need to create a tracks.txt file 
- This is how I randomized the order of the playlists to get the first 100 tracks in each playlist.
- Each playlist can be viewed on my spotify profile, and is detailed in a table on the Final Report

### 2. main.py
- This file will not be able to run because it would give access to my Spotify Account, and I've removed the ID and Secret used to make the API calls
- I'm including this file for your reference on how I obtained my data, which can be viewed in the track_features.csv file

### 3. explore.py
- This file runs the exploratory analysis for the genres and features, plotting histograms, kde plots, and boxplots
- This file also shuffles the data and prepares different subsets for the classification analysis in analysis.py

### 4. analysis.py
- This file runs the classification models, including PCA, and cross validation of the models
- This file also generates the confusion matrix plots and PCA accuracy plots in addition to the feature importance plot

