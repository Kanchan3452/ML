import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# 1. READ CSV FILE
# ==========================================================

df = pd.read_csv("faaah.csv")


# Remove accidental spaces from column names
df.columns = df.columns.str.strip()


# ==========================================================
# 2. KEEP ONLY REQUIRED COLUMNS
# ==========================================================

df = df[
    [
        "Year",
        "Element",
        "Value"
    ]
]


# Make sure Value is numeric
df["Value"] = pd.to_numeric(
    df["Value"],
    errors="coerce"
)


# Make sure Year is numeric
df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)


# ==========================================================
# 3. CONVERT DATA FROM LONG FORMAT TO WIDE FORMAT
# ==========================================================
#
# Original:
#
# Year   Element           Value
# 1961   Area harvested    375000
# 1961   Yield              7250.7
# 1961   Production       2719000
#
# Becomes:
#
# Year   Area harvested   Yield    Production
# 1961      375000        7250.7    2719000
#
# ==========================================================

data = df.pivot(
    index="Year",
    columns="Element",
    values="Value"
).reset_index()


# ==========================================================
# 4. RENAME COLUMNS
# ==========================================================

data = data.rename(
    columns={
        "Area harvested": "Area_Harvested",
        "Yield": "Yield",
        "Production": "Production"
    }
)


# ==========================================================
# 5. SORT DATA CHRONOLOGICALLY
# ==========================================================

data = data.sort_values(
    "Year"
).reset_index(drop=True)


# ==========================================================
# 6. REMOVE MISSING VALUES
# ==========================================================

data = data.dropna(
    subset=[
        "Year",
        "Area_Harvested",
        "Yield",
        "Production"
    ]
)


# ==========================================================
# 7. DISPLAY PREPARED DATA
# ==========================================================

print("\n================================================")
print("PREPARED DATA")
print("================================================")

print(
    data.head(10).to_string(index=False)
)


# ==========================================================
# 8. INPUT VARIABLES AND TARGET
# ==========================================================
#
# X1 = Year
# X2 = Area Harvested
# X3 = Yield
#
# Target = Production
#
# ==========================================================

X = data[
    [
        "Year",
        "Area_Harvested",
        "Yield"
    ]
]

y = data["Production"]


# ==========================================================
# 9. SEQUENTIAL TRAIN / TEST SPLIT
# ==========================================================
#
# IMPORTANT:
# We DO NOT randomly shuffle the data.
#
# First 80% = Training
# Last 20%  = Testing
#
# ==========================================================

split_index = int(
    len(data) * 0.80
)


X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]


X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]


# ==========================================================
# 10. PRINT TRAINING AND TESTING PERIOD
# ==========================================================

print("\n================================================")
print("TRAINING / TESTING")
print("================================================")

print(
    "Total data:",
    len(data)
)

print(
    "Training data:",
    len(X_train)
)

print(
    "Testing data:",
    len(X_test)
)

print(
    f"Training years: "
    f"{int(X_train['Year'].iloc[0])} - "
    f"{int(X_train['Year'].iloc[-1])}"
)

print(
    f"Testing years: "
    f"{int(X_test['Year'].iloc[0])} - "
    f"{int(X_test['Year'].iloc[-1])}"
)


# ==========================================================
# 11. CREATE MULTIVARIATE LINEAR REGRESSION MODEL
# ==========================================================

model = LinearRegression()


# ==========================================================
# 12. TRAIN MODEL
# ==========================================================

model.fit(
    X_train,
    y_train
)


# ==========================================================
# 13. GET MODEL PARAMETERS
# ==========================================================

b = model.intercept_

w1 = model.coef_[0]   # Year
w2 = model.coef_[1]   # Area harvested
w3 = model.coef_[2]   # Yield


# ==========================================================
# 14. DISPLAY MODEL PARAMETERS
# ==========================================================

print("\n================================================")
print("MULTIVARIATE LINEAR REGRESSION")
print("================================================")

print(
    f"Bias (b)               = {b:.6f}"
)

print(
    f"Weight (w1) - Year     = {w1:.6f}"
)

print(
    f"Weight (w2) - Area     = {w2:.6f}"
)

print(
    f"Weight (w3) - Yield    = {w3:.6f}"
)


# ==========================================================
# 15. HYPOTHESIS EQUATION
# ==========================================================

print("\n================================================")
print("HYPOTHESIS EQUATION")
print("================================================")

print(
    "h(X) = b + w1(Year) + w2(Area) + w3(Yield)"
)

print()

print(
    f"h(X) = "
    f"{b:.6f} "
    f"+ ({w1:.6f})Year "
    f"+ ({w2:.6f})Area "
    f"+ ({w3:.6f})Yield"
)


# ==========================================================
# 16. PREDICT TRAINING DATA
# ==========================================================

train_predictions = model.predict(
    X_train
)


# ==========================================================
# 17. PREDICT TEST DATA
# ==========================================================

test_predictions = model.predict(
    X_test
)


# ==========================================================
# 18. DISPLAY TEST PREDICTIONS
# ==========================================================

results = pd.DataFrame({

    "Year":
        X_test["Year"].values,

    "Actual Production":
        y_test.values,

    "Predicted Production":
        test_predictions,

    "Error":
        y_test.values - test_predictions

})


print("\n================================================")
print("TEST DATA: ACTUAL VS PREDICTED")
print("================================================")

print(
    results.to_string(index=False)
)


# ==========================================================
# 19. MODEL EVALUATION
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


# ==========================================================
# 20. DISPLAY PERFORMANCE
# ==========================================================

print("\n================================================")
print("MODEL PERFORMANCE")
print("================================================")

print(
    f"MAE  = {mae:,.4f}"
)

print(
    f"MSE  = {mse:,.4f}"
)

print(
    f"RMSE = {rmse:,.4f}"
)

print(
    f"R²   = {r2:.4f}"
)


# ==========================================================
# 21. FUTURE PREDICTION
# ==========================================================
#
# A multivariate model requires ALL input variables.
#
# Therefore, for future years we need:
#
# Year
# Area harvested
# Yield
#
# Since future Area and Yield are unknown,
# we use the latest known values as assumptions.
#
# ==========================================================

last_area = data[
    "Area_Harvested"
].iloc[-1]

last_yield = data[
    "Yield"
].iloc[-1]


future_data = pd.DataFrame({

    "Year": [
        2027,
        2028,
        2029,
        2030
    ],

    "Area_Harvested": [
        last_area,
        last_area,
        last_area,
        last_area
    ],

    "Yield": [
        last_yield,
        last_yield,
        last_yield,
        last_yield
    ]

})


# ==========================================================
# 22. PREDICT FUTURE PRODUCTION
# ==========================================================

future_predictions = model.predict(
    future_data
)


# ==========================================================
# 23. DISPLAY FUTURE PREDICTIONS
# ==========================================================

future_results = future_data.copy()

future_results[
    "Predicted Production"
] = future_predictions


print("\n================================================")
print("FUTURE PRODUCTION PREDICTIONS")
print("================================================")

print(
    future_results.to_string(index=False)
)


# ==========================================================
# 24. GRAPH 1
# ACTUAL VS PREDICTED PRODUCTION OVER TIME
# ==========================================================

plt.figure(
    figsize=(14, 7)
)


# Actual training data
plt.plot(
    X_train["Year"],
    y_train,
    color="blue",
    linewidth=2,
    label="Training Actual"
)


# Actual test data
plt.plot(
    X_test["Year"],
    y_test,
    color="black",
    linewidth=3,
    marker="o",
    markersize=5,
    label="Test Actual"
)


# Test predictions
plt.plot(
    X_test["Year"],
    test_predictions,
    color="red",
    linewidth=2,
    marker="x",
    markersize=7,
    linestyle="--",
    label="Test Predicted"
)


# Future predictions
plt.plot(
    future_data["Year"],
    future_predictions,
    color="green",
    linewidth=3,
    marker="o",
    markersize=7,
    linestyle="--",
    label="Future Prediction"
)


# Vertical line separating training and testing
plt.axvline(
    x=X_test["Year"].iloc[0],
    color="gray",
    linestyle=":",
    linewidth=2,
    label="Train/Test Boundary"
)


plt.xlabel(
    "Year",
    fontsize=12
)

plt.ylabel(
    "Potato Production (tonnes)",
    fontsize=12
)

plt.title(
    "Multivariate Regression: Potato Production Prediction",
    fontsize=15,
    fontweight="bold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ==========================================================
# 25. GRAPH 2
# ACTUAL VS PREDICTED SCATTER PLOT
# ==========================================================

plt.figure(
    figsize=(8, 8)
)


plt.scatter(
    y_test,
    test_predictions,
    color="red",
    s=80,
    edgecolor="black",
    label="Test Predictions"
)


# Perfect prediction line
minimum = min(
    y_test.min(),
    test_predictions.min()
)

maximum = max(
    y_test.max(),
    test_predictions.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="blue",
    linewidth=2,
    linestyle="--",
    label="Perfect Prediction"
)


plt.xlabel(
    "Actual Production (tonnes)",
    fontsize=12
)

plt.ylabel(
    "Predicted Production (tonnes)",
    fontsize=12
)

plt.title(
    "Actual vs Predicted Production",
    fontsize=14,
    fontweight="bold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()