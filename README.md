# Student Grade Predictor (Extended)

## Project Overview

Student Grade Predictor (Extended) is a Machine Learning project developed during an internship. It predicts students' academic performance using study hours, attendance, previous scores, and other factors.

The project gradually improves from basic Linear Regression to advanced techniques like Regularization and Visualization.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* VS Code
* Git & GitHub

---

## Dataset

The project uses student performance datasets containing features such as:

* Hours Studied
* Attendance
* Previous Scores
* Sleep Hours
* Motivation Level
* Tutoring Sessions
* Family Income
* Gender
* Exam Score

---

## Project Structure

student_grade_predictor/

├── charts/

│   ├── linear_actual_vs_predicted.png

│   ├── ridge_actual_vs_predicted.png

│   └── lasso_actual_vs_predicted.png

├── data/

│   ├── student_data.csv

│   ├── clean_student_data.csv

│   ├── rich_student_data.csv

│   └── spam.csv

├── src/

│   ├── data_cleaning.py

│   ├── linear_regression.py

│   ├── spam_classifier.py

│   ├── feature_engineering.py

│   ├── regularization.py

│   └── visualization.py

└── README.md

---

## Weekly Progress

### Week 1 – Data Cleaning

* Loaded dataset
* Removed duplicates
* Cleaned student data
* Saved clean dataset

### Week 2 – Linear Regression

* Trained Linear Regression model
* Predicted student grades
* Evaluated MAE, MSE, and R²

### Week 3 – Text Processing

* Built SMS Spam Classifier
* Applied TF-IDF
* Used Naive Bayes

### Week 4 – Feature Engineering

* Created richer student dataset
* Added engineered features
* Prepared dataset for better prediction

### Week 5 – Regularization

Compared:

* Linear Regression
* Ridge Regression
* Lasso Regression

Results:

* Ridge performed best with the lowest error.

### Week 6–7 – Visualization

Generated:

* Linear Regression Actual vs Predicted chart
* Ridge Regression Actual vs Predicted chart
* Lasso Regression Actual vs Predicted chart

Charts are saved inside the `charts` folder.

---

## Model Performance

| Model             |        MAE |        MSE |   R² Score |
| ----------------- | ---------: | ---------: | ---------: |
| Linear Regression |     0.4503 |     3.2567 |     0.7696 |
| Ridge Regression  | **0.4501** | **3.2563** | **0.7696** |
| Lasso Regression  |     0.9789 |     4.1456 |     0.7067 |

---

## How to Run

Clone the repository.

Run:

```bash
python src/regularization.py
```

For visualization:

```bash
python src/visualization.py
```

The generated graphs will be saved inside the `charts` folder.

---

## Future Improvements

* Hyperparameter tuning
* More advanced ML models
* Web interface using Flask or Streamlit
* Better feature selection