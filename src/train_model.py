import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/clean_student_data.csv")

print(df.head())

print(df.columns)

X = df[['Attendance', 'StudyHours', 'Assignments', 'PreviousGPA']]

grade_mapping = {
    'A': 4,
    'B': 3,
    'C': 2,
    'D': 1
}

df['FinalGrade'] = df['FinalGrade'].map(grade_mapping)

y = df['FinalGrade']


print(y.head())


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data :", X_test.shape)


model = LinearRegression()


model.fit(X_train, y_train)

print("Model trained successfully!")


y_pred = model.predict(X_test)

print("Actual Grades:")
print(y_test.values)

print("Predicted Grades:")
print(y_pred)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error :", mse)
print("R2 Score:", r2)