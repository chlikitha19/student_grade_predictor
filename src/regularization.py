import pandas as pd

df = pd.read_csv("data/student_performance_cleaned.csv")

print(df.head())

print(df.columns)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.get_dummies(df, drop_first=True)


X = df.drop("Exam_Score", axis=1)


y = df["Exam_Score"]

print("Features:", X.shape)
print("Target:", y.shape)
print("Features:", X.shape)
print("Target:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

print("Linear Regression model trained successfully!")

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

print("Ridge Regression model trained successfully!")

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

lasso_pred = lasso_model.predict(X_test)

print("Lasso Regression model trained successfully!")

def evaluate(name, y_true, y_pred):
  print(f"\n{name}")
  print("MAE:", mean_absolute_error(y_true, y_pred))
  print("MSE:", mean_squared_error(y_true, y_pred))
  print("R2 Score:", r2_score(y_true, y_pred))

evaluate("Linear Regression", y_test, linear_pred)
evaluate("Ridge Regression", y_test, ridge_pred)
evaluate("Lasso Regression", y_test, lasso_pred)