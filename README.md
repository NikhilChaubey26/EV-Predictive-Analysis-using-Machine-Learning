# EV Predictive Analysis using Machine Learning 🚗⚡

## Overview

This project focuses on analyzing electric vehicle (EV) population data using machine learning techniques. With the rapid growth of electric vehicles and increasing concerns about pollution and sustainability, large EV datasets are being generated. Manual analysis of such data is difficult, so machine learning models are used to extract insights and make accurate predictions.

The project includes classification, regression, and clustering tasks, along with an interactive Streamlit dashboard for visualization and prediction.

---

## Objectives

The main goals of this project are:

* To clean and preprocess real-world electric vehicle data
* To predict Clean Alternative Fuel Vehicle (CAFV) eligibility using classification models
* To estimate electric driving range using regression models
* To group similar electric vehicles using clustering techniques
* To build an interactive Streamlit dashboard for easy analysis and prediction

---

## Dataset

The dataset contains electric vehicle population data, including details such as vehicle make, model year, electric range, location, and CAFV eligibility. Before applying machine learning models, the data is cleaned to handle missing values, duplicates, and inconsistent columns.

---

## Data Preprocessing

The following preprocessing steps are performed:

* Column name cleaning and formatting
* Removal of duplicate records
* Handling missing values
* Dropping unnecessary identifier columns
* Encoding categorical variables using one-hot encoding
* Feature scaling where required
* Train-test split with stratification for classification tasks

---

## Machine Learning Models

### Classification (CAFV Eligibility Prediction)

* Logistic Regression
* Decision Tree
* Random Forest

### Regression (Electric Range Estimation)

* Linear Regression
* Random Forest Regressor

### Clustering

* K-Means Clustering to group electric vehicles with similar characteristics

Model performance is evaluated using metrics such as accuracy, ROC-AUC score, MAE, RMSE, and R² score.

---

## Streamlit Dashboard

A Streamlit web application is developed to:

* Visualize electric vehicle data and trends
* Predict CAFV eligibility based on user input
* Estimate electric driving range
* Display clustering results interactively

The dashboard makes the project user-friendly and accessible even to non-technical users.

---

## Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* Streamlit

---

## Project Structure

```
├── data/
│   └── ev_data.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   └── modeling.ipynb
├── app.py
├── requirements.txt
├── README.md
```

---

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/your-username/ev-predictive-analysis.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Run the Streamlit application:

```
streamlit run app.py
```

---

## Learning Outcomes

This project helped me gain practical experience in:

* End-to-end machine learning workflows
* Working with real-world datasets
* Feature engineering and model evaluation
* Building and deploying ML applications using Streamlit
* Applying machine learning in sustainability and clean energy domains

---

## Future Enhancements

* Use advanced models such as XGBoost or CatBoost
* Improve dashboard design and performance
* Add more visualizations and insights
* Deploy the application online

---

## Author

**Nikhil Chaubey**
Computer Science and Engineering
Aspiring Data Scien
