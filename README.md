EEG-Classification
==============================

This project focuses on the classification of gait patterns using raw EEG signals recorded from patients who underwent various gait trials. The gait trials were meticulously captured using the VICON System to extract precise gait patterns, while simultaneous recordings of brainwaves were obtained through EEG. The overarching goal is to develop a machine learning model capable of predicting gait patterns based on EEG time series data.
Both multiclass and single class segmentation is implemented

To run the Model, also individual arguments are able to beeing passed.
```
src/models/train_model.py
```

## Introduction:
Gait analysis is a critical component of assessing motor function and neurological disorders. Traditional methods involve the use of motion capture systems like VICON to precisely capture and analyze gait patterns. In this project, we extend these conventional approaches by incorporating EEG recordings to investigate the relationship between brain activity and gait .The precise synchronization of EEG and gait event data provides a rich dataset for training and validating machine learning models. This integrated approach contributes to a comprehensive understanding of the neural underpinnings of gait, potentially leading to advancements in personalized rehabilitation strategies, early detection of neurological disorders, and the development of neuroprosthetics. The findings from this study have broader implications for both clinical and research applications in the fields of neuroscience, biomechanics, and rehabilitation.


## Data Collection:
Participants underwent diverse gait trials, capturing a range of movements and scenarios, utilizing the VICON System for accurate gait pattern extraction.
Simultaneous EEG recordings were obtained to capture the corresponding brainwave activity during each gait trial.
The VICON data was used to extract the gaitevents: "Heel Strike", "Toe Off" to determine a timewindow for the 


## EEG Data: Raw EEG signals were preprocessed to remove noise, artifacts, and baseline drift.
Synchronization: Temporal synchronization between gait data and EEG recordings was ensured to align the two modalities accurately.



## Models: 
Implemented machine learning models, Transformers, Convolutional Neural Networks and long short-term memory networks (LSTMs), capable of learning temporal dependencies in EEG data.
The best performing model was the LSTM

## Future Analyis:
Instead of the time space it would be nice to explore the frequency domain to extract features to see which one correlate with what Gaitpattern as the Models were quite good for binary classifications but had drawbacks in multiclass classification.



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
