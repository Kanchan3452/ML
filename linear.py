import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# 1. READ CSV FILE
# ==========================================================

df = pd.read_csv("CHERRIES.csv")

# Make sure data is in sequential Year order
df = df.sort_values("Year").reset_index(drop=True)

print("Complete Dataset:")
print(df)


# ==========================================================
# 2. SELECT INPUT (X) AND OUTPUT (y)
# ==========================================================

X = df[['Year']]
y = df['Value']


# ==========================================================
# 3. SEQUENTIAL TRAIN-TEST SPLIT
# ==========================================================
# First 80%  -> Training data
# Last 20%   -> Testing data

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
# 4. TRAIN LINEAR REGRESSION USING TRAINING DATA ONLY
# ==========================================================

model = LinearRegression()

model.fit(X_train, y_train)


# ==========================================================
# 5. GET w AND b FROM TRAINING DATA
# ==========================================================

w = model.coef_[0]
b = model.intercept_

print("\n================================")
print("MODEL PARAMETERS")
print("================================")

print(f"Weight (w) = {w:.6f}")
print(f"Bias   (b) = {b:.6f}")


# ==========================================================
# 6. CREATE HYPOTHESIS EQUATION
# ==========================================================

print("\n================================")
print("HYPOTHESIS EQUATION")
print("================================")

print("h(w,b) = wX + b")

print(f"h(w,b) = {w:.6f}X + {b:.6f}")

print("\nTherefore:")

if b >= 0:
    print(f"Predicted Yield = {w:.6f} × Year + {b:.6f}")
else:
    print(f"Predicted Yield = {w:.6f} × Year - {abs(b):.6f}")


# ==========================================================
# 7. USE THE TRAINING EQUATION ON TEST DATA
# ==========================================================
# IMPORTANT:
# We do NOT train the model again.
# We use the w and b obtained from training data.

test_predictions = w * X_test['Year'] + b


# ==========================================================
# 8. DISPLAY TEST RESULTS
# ==========================================================

test_results = pd.DataFrame({
    "Year": X_test['Year'].values,
    "Actual Yield": y_test.values,
    "Predicted Yield": test_predictions.values
})

test_results["Error"] = (
    test_results["Actual Yield"]
    - test_results["Predicted Yield"]
)

print("\n================================")
print("TEST DATA: ACTUAL vs PREDICTED")
print("================================")

print(test_results.to_string(index=False))


# ==========================================================
# 9. CALCULATE TEST ERROR
# ==========================================================

mae = mean_absolute_error(y_test, test_predictions)

mse = mean_squared_error(y_test, test_predictions)

rmse = mse ** 0.5

r2 = r2_score(y_test, test_predictions)


print("\n================================")
print("TEST PERFORMANCE")
print("================================")

print(f"MAE  = {mae:.4f}")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")


# ==========================================================
# 10. PREDICT FUTURE YEARS USING TRAINING EQUATION
# ==========================================================
# Here we use the SAME w and b obtained from training data.

future_years = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030]
})

future_predictions = (
    w * future_years["Year"] + b
)


print("\n================================")
print("FUTURE PREDICTIONS")
print("================================")

for year, prediction in zip(
        future_years["Year"],
        future_predictions):

    print(
        f"{year}: {prediction:.2f}"
    )


# ==========================================================
# 11. PLOT TRAINING, TESTING AND FUTURE DATA
# ==========================================================

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


# Regression line obtained from TRAINING DATA
plt.plot(
    X,
    w * X["Year"] + b,
    color="red",
    label="Regression Line"
)


# Future predictions
plt.scatter(
    future_years,
    future_predictions,
    color="green",
    marker="x",
    s=100,
    label="Future Predictions"
)


plt.xlabel("Year")
plt.ylabel("Yield")

plt.title(
    "Crop Yield Prediction using Linear Regression"
)

plt.legend()
plt.grid(True)

plt.show()