Airbnb Rental Price Prediction (London)

This project focuses on predicting Airbnb rental prices in London using machine learning techniques.
The system includes data exploration, model training, and an interactive Streamlit web application with map-based visualization.

Project Overview

The big growth of short-term rental platforms such as Airbnb has made pricing optimization a key challenge for hosts.
When travelling to UK became to expsensive people need to calculate how much money they will spend before hand and this prediction model helps them to overcome the problem of 
findind approximate price of accomodation.

Key features:

Exploratory Data Analysis (EDA)

Machine Learning model comparison

Final model training and evaluation

Interactive Streamlit application with maps

End-to-end reproducible pipeline

📊 Dataset

The dataset contains Airbnb listings in London with features such as:

* Latitude & Longitude

* Neighbourhood

* Room type

* Reviews and availability

* Price per night

****Note:
Due to GitHub file size limitations, the dataset is not included in this repository.

Dataset Source
 (https://www.kaggle.com/datasets/thedevastator/airbnb-listings-in-london-neighbourhoods-and-rev?utm_source=chatgpt.com&select=listings.csv)
 (Airbnb London dataset)

Exploratory Data Analysis

EDA is performed in the notebook:

notebooks/cleaning.ipynb


It includes:

*Missing value analysis

*Price distribution and log-transformation

*Geographic visualization

*Feature relationships and correlations

Screenshots from this notebook are used in the final report.

Model Training

Several models were evaluated:

Ridge Regression (baseline)

RandomForestRegressor (final model)

The RandomForestRegressor achieved the best performance and was selected as the final model.

Evaluation Metrics

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Train the model

From the project root directory:

python src/train.py


This will:

preprocess the data

train the model

save the trained pipeline to:

models/model.joblib
The trained model file is not stored in GitHub due to size constraints.

Streamlit Web Application

The project includes an interactive Streamlit app with:

Map visualization of Airbnb listings

User input form for price prediction

Real-time model inference

Run the app locally
python -m streamlit run app/Home.py


The app will open at:

http://localhost:8501

Project Structure
housing/
├── app/
│   └── Home.py              # Streamlit application
├── src/
│   ├── train.py             # Model training script
│   └── predict.py           # Prediction utilities
├── notebooks/
│   └── 01_eda.ipynb         # Exploratory Data Analysis
├── data/
│   └── README.md            # Instructions for dataset
├── models/
│   └── README.md            # Instructions for model generation
├── requirements.txt
├── .gitignore
└── README.md

Requirements

Install dependencies using:

pip install -r requirements.txt

Main libraries used:

pandas

numpy

scikit-learn

joblib

streamlit

Large Files & GitHub Limitations

The following files are intentionally excluded:

data/*.csv

models/*.joblib

This ensures:

compliance with GitHub size limits

clean and reproducible repository

All results can be regenerated using the provided scripts.

https://youtu.be/BF_sGvn_duc?si=nt9vFJdJLNtZ0pr5

A short screen-recorded demo (3–5 minutes) demonstrating:

project overview

model training

Streamlit application usage

Iskander Manat GH1034435
2025
