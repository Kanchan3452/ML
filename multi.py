import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# 1. READ CSV FILE
# ==========================================================

df = pd.read_csv("FAOSTAT_data_en_8-22-2026.csv")


# ==========================================================
# 2. SELECT MAIZE DATA FOR INDIA
# ==========================================================

df = df[
    (df["Area"] == "India") &
    (df["Item"] == "Maize (corn)")
].copy()


# ==========================================================
# 3. CONVERT DATA INTO ONE ROW PER YEAR
# ==========================================================

df = df.pivot(
    index="Year",
    columns="Element",
    values="Value"
).reset_index()


# Sort by Year
df = df.sort_values("Year").reset_index(drop=True)


print("Complete Dataset:")
print(
    df[
        ["Year", "Area harvested", "Yield"]
    ].to_string(index=False)
)


# ==========================================================
# 4. SELECT MULTIPLE INPUT VARIABLES
# ==========================================================

# X1 = Year
# X2 = Area harvested
#
# Y = Yield

X = df[
    ["Year", "Area harvested"]
]

y = df["Yield"]


# ==========================================================
# 5. 80% TRAINING - 20% TESTING
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

print(
    pd.DataFrame({
        "Year": X_train["Year"],
        "Area Harvested": X_train["Area harvested"],
        "Yield": y_train
    })
    .sort_values("Year")
    .to_string(index=False)
)


print("\n================================")
print("TESTING DATA")
print("================================")

print(
    pd.DataFrame({
        "Year": X_test["Year"],
        "Area Harvested": X_test["Area harvested"],
        "Yield": y_test
    })
    .sort_values("Year")
    .to_string(index=False)
)


# ==========================================================
# 6. TRAIN MULTIVARIATE LINEAR REGRESSION
# ==========================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# ==========================================================
# 7. GET MODEL PARAMETERS
# ==========================================================

b = model.intercept_

w1 = model.coef_[0]

w2 = model.coef_[1]


print("\n================================")
print("MODEL PARAMETERS")
print("================================")

print(f"Bias (b) = {b:.6f}")

print(f"Weight for Year (w1) = {w1:.6f}")

print(
    f"Weight for Area Harvested (w2) = "
    f"{w2:.6f}"
)


# ==========================================================
# 8. HYPOTHESIS EQUATION
# ==========================================================

print("\n================================")
print("HYPOTHESIS EQUATION")
print("================================")

print("h(w) = b + w1X1 + w2X2")

print(
    f"h(w) = {b:.6f} "
    f"+ ({w1:.6f}) × Year "
    f"+ ({w2:.6f}) × Area Harvested"
)


# ==========================================================
# 9. PREDICT TEST DATA
# ==========================================================

test_predictions = model.predict(
    X_test
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
# 12. FUTURE PREDICTION
# ==========================================================

# For future prediction we need:
# Year AND Area Harvested

future_data = pd.DataFrame({
    "Year": [2027, 2028, 2029, 2030],

    "Area harvested": [
        5500000,
        5550000,
        5600000,
        5650000
    ]
})


future_predictions = model.predict(
    future_data
)


print("\n================================")
print("FUTURE PREDICTIONS")
print("================================")


for year, prediction in zip(
        future_data["Year"],
        future_predictions):

    print(
        f"{year}: {prediction:.2f} kg/ha"
    )


# ==========================================================
# 13. GRAPH
# ==========================================================

plt.figure(figsize=(10, 6))


# Actual training data
plt.scatter(
    X_train["Year"],
    y_train,
    color="blue",
    label="Training Data"
)


# Actual testing data
plt.scatter(
    X_test["Year"],
    y_test,
    color="orange",
    label="Actual Test Data"
)


# Test predictions
plt.scatter(
    X_test["Year"],
    test_predictions,
    color="red",
    marker="x",
    s=80,
    label="Test Predictions"
)


# Future predictions
plt.scatter(
    future_data["Year"],
    future_predictions,
    color="green",
    marker="*",
    s=150,
    label="Future Predictions"
)


plt.xlabel("Year")

plt.ylabel("Maize Yield (kg/ha)")

plt.title(
    "Maize Yield Prediction using Multivariate Linear Regression"
)

plt.legend()

plt.grid(True)

plt.show()