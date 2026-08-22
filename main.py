import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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


# Make sure data is sorted according to Year

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
# 5. TRAIN LINEAR REGRESSION
# ==========================================================

model = LinearRegression()

model.fit(X_train, y_train)


# ==========================================================
# 6. GET WEIGHT (w) AND BIAS (b)
# ==========================================================

w = model.coef_[0]
b = model.intercept_


print("\n================================")
print("MODEL PARAMETERS")
print("================================")

print(f"Weight (w) = {w:.6f}")
print(f"Bias   (b) = {b:.6f}")


# ==========================================================
# 7. HYPOTHESIS EQUATION
# ==========================================================

print("\n================================")
print("HYPOTHESIS EQUATION")
print("================================")

print("h(w,b) = wX + b")

print(f"h(w,b) = {w:.6f}X + {b:.6f}")

print("\nTherefore:")

if b >= 0:
    print(
        f"Predicted Maize Yield = "
        f"{w:.6f} × Year + {b:.6f}"
    )
else:
    print(
        f"Predicted Maize Yield = "
        f"{w:.6f} × Year - {abs(b):.6f}"
    )


# ==========================================================
# 8. PREDICT TEST DATA
# ==========================================================

test_predictions = model.predict(X_test)


# ==========================================================
# 9. DISPLAY TEST RESULTS
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
# 10. CALCULATE TEST PERFORMANCE
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
# 11. FUTURE PREDICTIONS
# ==========================================================

future_years = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030]
})


future_predictions = model.predict(
    future_years
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
# 12. PLOT TRAINING, TESTING AND FUTURE DATA
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


# Regression line
plt.plot(
    X["Year"],
    model.predict(X),
    color="red",
    label="Regression Line"
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
# 13. GRAPH LABELS
# ==========================================================

plt.xlabel("Year")

plt.ylabel("Maize Yield (kg/ha)")

plt.title(
    "Maize Yield Prediction using Linear Regression"
)

plt.legend()

plt.grid(True)

plt.show()