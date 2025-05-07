import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('StudentsPerformance.csv')

df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

df = df[['math_score', 'reading_score', 'writing_score']].apply(pd.to_numeric)

print("Estadísticas descriptivas:")
print(df.describe())

scatter_matrix(df, figsize=(8, 6))
plt.suptitle("Matriz de dispersión entre puntajes")
plt.tight_layout()
plt.show()

X = df[['reading_score', 'writing_score']]
y = df['math_score']

for split in [(0.7, 0.3), (0.5, 0.5), (0.4, 0.6)]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split[0], random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\nSplit {int(split[0]*100)}-{int(split[1]*100)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train, y_train)
y_pred_sgd = sgd.predict(X_test)
print("\nSGD Regressor")
print(f"MSE: {mean_squared_error(y_test, y_pred_sgd):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_sgd):.2f}")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("\nRidge Regularization")
print(f"MSE: {mean_squared_error(y_test, y_pred_ridge):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_ridge):.2f}")

print("\nLasso Regularization")
print(f"MSE: {mean_squared_error(y_test, y_pred_lasso):.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred_lasso):.2f}")

final_model = Ridge(alpha=1.0)
final_model.fit(X_train, y_train)

print("\nModelo Final (Ridge, 70/30)")
print(f"Intercepto: {final_model.intercept_:.2f}")
print(f"Coeficientes: {final_model.coef_}")

final_pred = final_model.predict(X_test)
print(f"MSE Final: {mean_squared_error(y_test, final_pred):.2f}")
print(f"R2 Score Final: {r2_score(y_test, final_pred):.2f}")

