# ==========================================================
# POLYNOMIAL REGRESSION FOR CROP YIELD PREDICTION
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# 1. READ CSV FILE
# ==========================================================

df = pd.read_csv("CHERRIES.csv")

# Make sure data is arranged in Year order
df = df.sort_values("Year").reset_index(drop=True)

print("Complete Dataset:")
print(df)


# ==========================================================
# 2. SELECT INPUT AND OUTPUT
# ==========================================================

# X = Year
# y = Crop Yield / Value

X = df[['Year']]
y = df['Value']


# ==========================================================
# 3. SEQUENTIAL TRAIN-TEST SPLIT
# ==========================================================

# First 80% = Training
# Last 20%  = Testing

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]


print("\n================================")
print("TRAINING DATA")
print("================================")

print(pd.DataFrame({
    "Year": X_train["Year"],
    "Value": y_train
}))


print("\n================================")
print("TESTING DATA")
print("================================")

print(pd.DataFrame({
    "Year": X_test["Year"],
    "Value": y_test
}))


# ==========================================================
# 4. CREATE POLYNOMIAL FEATURES
# ==========================================================

# Degree = 2 means:
#
# X becomes:
#
# X, X²
#
# Therefore the equation becomes:
#
# h(X) = b + w1X + w2X²

degree = 2

poly = PolynomialFeatures(degree=degree)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)


# ==========================================================
# 5. TRAIN LINEAR REGRESSION ON POLYNOMIAL FEATURES
# ==========================================================

model = LinearRegression()

model.fit(X_train_poly, y_train)


# ==========================================================
# 6. GET THE PARAMETERS
# ==========================================================

# For degree 2:
#
# h(X) = b + w1X + w2X²

b = model.intercept_

w1 = model.coef_[1]
w2 = model.coef_[2]


print("\n================================")
print("POLYNOMIAL MODEL PARAMETERS")
print("================================")

print(f"Bias (b)       = {b:.6f}")
print(f"Weight (w1)    = {w1:.6f}")
print(f"Weight (w2)    = {w2:.6f}")


# ==========================================================
# 7. DISPLAY HYPOTHESIS EQUATION
# ==========================================================

print("\n================================")
print("HYPOTHESIS EQUATION")
print("================================")

print("h(X) = b + w1X + w2X²")

print(
    f"h(X) = {b:.6f} "
    f"+ ({w1:.6f})X "
    f"+ ({w2:.6f})X²"
)


# ==========================================================
# 8. PREDICT TEST DATA USING THE EQUATION
# ==========================================================

# First create X² for test data.
#
# Then:
#
# h(X) = b + w1X + w2X²

test_predictions = (
    b
    + w1 * X_test["Year"]
    + w2 * (X_test["Year"] ** 2)
)


# ==========================================================
# 9. DISPLAY ACTUAL VS PREDICTED TEST VALUES
# ==========================================================

test_results = pd.DataFrame({
    "Year": X_test["Year"].values,
    "Actual Yield": y_test.values,
    "Predicted Yield": test_predictions.values
})

test_results["Error"] = (
    test_results["Actual Yield"]
    - test_results["Predicted Yield"]
)


print("\n================================")
print("TEST RESULTS")
print("================================")

print(test_results.to_string(index=False))


# ==========================================================
# 10. TEST PERFORMANCE
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
print("MODEL PERFORMANCE")
print("================================")

print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")


# ==========================================================
# 11. FUTURE PREDICTION
# ==========================================================

future_years = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030]
})


# Apply the SAME equation:
#
# h(X) = b + w1X + w2X²

future_predictions = (
    b
    + w1 * future_years["Year"]
    + w2 * (future_years["Year"] ** 2)
)


print("\n================================")
print("FUTURE PREDICTIONS")
print("================================")

for year, prediction in zip(
        future_years["Year"],
        future_predictions):

    print(f"{year}: {prediction:.2f}")


# ==========================================================
# 12. PLOT
# ==========================================================

# Create smooth years for drawing the curve

plot_years = pd.DataFrame({
    "Year": range(
        int(X["Year"].min()),
        2031
    )
})


plot_predictions = (
    b
    + w1 * plot_years["Year"]
    + w2 * (plot_years["Year"] ** 2)
)


plt.figure(figsize=(10, 6))


# Training data
plt.scatter(
    X_train,
    y_train,
    color="blue",
    label="Training Data"
)


# Actual test data
plt.scatter(
    X_test,
    y_test,
    color="orange",
    label="Actual Test Data"
)


# Polynomial curve
plt.plot(
    plot_years["Year"],
    plot_predictions,
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


plt.xlabel("Year")
plt.ylabel("Yield")

plt.title(
    "Crop Yield Prediction using Polynomial Regression"
)

plt.legend()
plt.grid(True)

plt.show()
