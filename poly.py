import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# 1. READ CSV FILE
# ==========================================================

df = pd.read_csv("FAOSTAT_data_en_8-22-2026.csv")


# ==========================================================
# 2. SELECT MAIZE YIELD DATA FOR INDIA
# ==========================================================

df = df[
    (df["Area"] == "India") &
    (df["Item"] == "Maize (corn)") &
    (df["Element"] == "Yield")
].copy()


# Sort according to Year
df = df.sort_values("Year").reset_index(drop=True)


print("Complete Dataset:")
print(df[["Year", "Value"]].to_string(index=False))


# ==========================================================
# 3. SELECT INPUT (X) AND OUTPUT (y)
# ==========================================================

X = df[["Year"]]
y = df["Value"]


# ==========================================================
# 4. 80% TRAINING - 20% TESTING
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)


print("\n================================")
print("TRAINING DATA")
print("================================")

print(pd.DataFrame({
    "Year": X_train["Year"],
    "Yield": y_train
}).sort_values("Year").to_string(index=False))


print("\n================================")
print("TESTING DATA")
print("================================")

print(pd.DataFrame({
    "Year": X_test["Year"],
    "Yield": y_test
}).sort_values("Year").to_string(index=False))


# ==========================================================
# 5. CREATE POLYNOMIAL FEATURES
# ==========================================================

degree = 2

poly = PolynomialFeatures(
    degree=degree
)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)


# ==========================================================
# 6. TRAIN POLYNOMIAL REGRESSION MODEL
# ==========================================================

model = LinearRegression()

model.fit(
    X_train_poly,
    y_train
)


# ==========================================================
# 7. GET MODEL COEFFICIENTS
# ==========================================================

b0 = model.intercept_
b1 = model.coef_[1]
b2 = model.coef_[2]


print("\n================================")
print("MODEL PARAMETERS")
print("================================")

print(f"Bias (b0) = {b0:.6f}")
print(f"Coefficient (b1) = {b1:.6f}")
print(f"Coefficient (b2) = {b2:.6f}")


# ==========================================================
# 8. POLYNOMIAL HYPOTHESIS EQUATION
# ==========================================================

print("\n================================")
print("HYPOTHESIS EQUATION")
print("================================")

print("h(x) = b0 + b1X + b2X²")

print(
    f"h(x) = {b0:.6f} "
    f"+ ({b1:.6f})X "
    f"+ ({b2:.6f})X²"
)


# ==========================================================
# 9. PREDICT TEST DATA
# ==========================================================

test_predictions = model.predict(
    X_test_poly
)


# ==========================================================
# 10. DISPLAY TEST RESULTS
# ==========================================================

test_results = pd.DataFrame({
    "Year": X_test["Year"].values,
    "Actual Yield": y_test.values,
    "Predicted Yield": test_predictions
})

test_results["Error"] = (
    test_results["Actual Yield"]
    - test_results["Predicted Yield"]
)


print("\n================================")
print("TEST DATA: ACTUAL vs PREDICTED")
print("================================")

print(
    test_results
    .sort_values("Year")
    .to_string(index=False)
)


# ==========================================================
# 11. CALCULATE TEST PERFORMANCE
# ==========================================================

mae = mean_absolute_error(
    y_test,
    test_predictions
)

mse = mean_squared_error(
    y_test,
    test_predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    test_predictions
)


print("\n================================")
print("TEST PERFORMANCE")
print("================================")

print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")


# ==========================================================
# 12. FUTURE PREDICTIONS
# ==========================================================

future_years = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030]
})


future_years_poly = poly.transform(
    future_years
)


future_predictions = model.predict(
    future_years_poly
)


print("\n================================")
print("FUTURE PREDICTIONS")
print("================================")

for year, prediction in zip(
        future_years["Year"],
        future_predictions):

    print(
        f"{year}: {prediction:.2f} kg/ha"
    )


# ==========================================================
# 13. CREATE VALUES FOR REGRESSION CURVE
# ==========================================================

X_curve = pd.DataFrame({
    "Year": range(
        int(df["Year"].min()),
        2031
    )
})


X_curve_poly = poly.transform(
    X_curve
)


curve_predictions = model.predict(
    X_curve_poly
)


# ==========================================================
# 14. PLOT GRAPH
# ==========================================================

plt.figure(figsize=(10, 6))


# Training data
plt.scatter(
    X_train["Year"],
    y_train,
    color="blue",
    label="Training Data"
)


# Actual test data
plt.scatter(
    X_test["Year"],
    y_test,
    color="orange",
    label="Actual Test Data"
)


# Polynomial regression curve
plt.plot(
    X_curve["Year"],
    curve_predictions,
    color="red",
    label="Polynomial Regression Curve"
)


# Future predictions
plt.scatter(
    future_years["Year"],
    future_predictions,
    color="green",
    marker="x",
    s=100,
    label="Future Predictions"
)


# ==========================================================
# 15. GRAPH LABELS
# ==========================================================

plt.xlabel("Year")

plt.ylabel("Maize Yield (kg/ha)")

plt.title(
    "Maize Yield Prediction using Polynomial Regression"
)

plt.legend()

plt.grid(True)

plt.show()