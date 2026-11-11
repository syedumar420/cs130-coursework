# AI-Based Student Performance Prediction System

A simple student mini project using Machine Learning and Flask.

## Features
- Takes student academic/lifestyle inputs
- Predicts `Good` or `Needs Improvement`
- Displays prediction confidence
- Random Forest classifier
- Simple Flask web interface
- Includes a sample CSV dataset

## Inputs
1. Attendance (%)
2. Study Hours per Day
3. Previous Marks (%)
4. Assignment Score (%)
5. Internal Marks (%)
6. Sleep Hours per Day

## Technology
- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- HTML/CSS

## Windows Run Commands

Open Command Prompt inside the project folder.

### 1. Create virtual environment
python -m venv venv

### 2. Activate it
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Train the ML model
python train_model.py

### 5. Run Flask application
python app.py

### 6. Open in browser
http://127.0.0.1:5000/

## Quick Run (without virtual environment)
pip install -r requirements.txt
python train_model.py
python app.py

## Important
This is an educational demonstration project. The included dataset is synthetic/sample data and predictions should not be used for real academic decisions.
