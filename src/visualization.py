import pandas as pd

df = pd.read_csv("data/student_performance_cleaned.csv")

print(df.head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Exam_Score", axis=1)
y = df["Exam_Score"]
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)
plt.figure(figsize=(6,6))

plt.scatter(y_test, linear_pred)

plt.plot(
[y_test.min(), y_test.max()],
[y_test.min(), y_test.max()]
)

plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Linear Regression: Actual vs Predicted")

plt.savefig("charts/linear_actual_vs_predicted.png")

plt.show()
plt.figure(figsize=(6,6))

plt.scatter(y_test, ridge_pred)
plt.plot(
[y_test.min(), y_test.max()],
[y_test.min(), y_test.max()],
color="red"
)

plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Ridge Regression: Actual vs Predicted")

plt.savefig("charts/ridge_actual_vs_predicted.png")
plt.show()
plt.figure(figsize=(6,6))

plt.scatter(y_test, lasso_pred)
plt.plot(
[y_test.min(), y_test.max()],
[y_test.min(), y_test.max()],
color="red"
)

plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Lasso Regression: Actual vs Predicted")

plt.savefig("charts/lasso_actual_vs_predicted.png")
plt.show()