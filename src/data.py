import pandas as pd
import config


df = pd.read_csv(config.TRAINING_FILE)
percent_missing = df.isnull().sum() * 100 / len(df)

# check percentage of cols null values and filter out 25 or over
for idx, value in enumerate(percent_missing):
    if value > 25:
        print(idx)
df = df.drop(["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"], axis=1)

# Date features
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# dropp features and null-values
df = df.drop(
    [
        "Date",
        "Location",
        "WindGustDir",
        "WindGustSpeed",
        "WindDir9am",
        "WindDir3pm",
        "WindSpeed9am",
        "WindSpeed3pm",
        "WindGustDir",
    ],
    axis=1,
)
df["RainToday"].replace({"No": 0, "Yes": 1}, inplace=True)
df["RainTomorrow"].replace({"No": 0, "Yes": 1}, inplace=True)
df = df.dropna()

if df.isnull().sum().any() == False:
    print("Data is Clean")
    df.to_csv(config.CLEAN_FILE, index=False)

else:
    print("Data is not clean")
