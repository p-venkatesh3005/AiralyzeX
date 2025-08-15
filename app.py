from flask import Flask, render_template
import pandas as pd
from prophet import Prophet
import os

app = Flask(__name__)

def compute_aqi(row):
    pm25 = row["pm25_value"]
    if pm25 <= 50:
        return "Good"
    elif pm25 <= 100:
        return "Moderate"
    elif pm25 <= 200:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

def generate_forecast():
    df = pd.read_csv("pol_data.csv")
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)

    # NO2
    no2_df = df[["datetime", "no2_value"]].rename(columns={"datetime": "ds", "no2_value": "y"})
    no2_df["y"] = no2_df["y"].fillna(no2_df["y"].mean())
    model_no2 = Prophet()
    model_no2.fit(no2_df)
    future_no2 = model_no2.make_future_dataframe(periods=12, freq="6H")
    last_date_no2 = no2_df["ds"].max()
    forecast_no2 = model_no2.predict(future_no2)
    forecast_no2 = forecast_no2[forecast_no2["ds"] > last_date_no2]
    forecast_no2 = forecast_no2[["ds", "yhat"]].rename(columns={"yhat": "NO2"})

    # SO2
    so2_df = df[["datetime", "so2_value"]].rename(columns={"datetime": "ds", "so2_value": "y"})
    so2_df["y"] = so2_df["y"].fillna(so2_df["y"].mean())
    model_so2 = Prophet()
    model_so2.fit(so2_df)
    future_so2 = model_so2.make_future_dataframe(periods=12, freq="6H")
    last_date_so2 = so2_df["ds"].max()
    forecast_so2 = model_so2.predict(future_so2)
    forecast_so2 = forecast_so2[forecast_so2["ds"] > last_date_so2]
    forecast_so2 = forecast_so2[["ds", "yhat"]].rename(columns={"yhat": "SO2"})

    # PM2.5
    pm25_df = df[["datetime", "pm25_value"]].rename(columns={"datetime": "ds", "pm25_value": "y"})
    pm25_df["y"] = pm25_df["y"].fillna(pm25_df["y"].mean())
    model_pm25 = Prophet()
    model_pm25.fit(pm25_df)
    future_pm25 = model_pm25.make_future_dataframe(periods=12, freq="6H")
    last_date_pm25 = pm25_df["ds"].max()
    forecast_pm25 = model_pm25.predict(future_pm25)
    forecast_pm25 = forecast_pm25[forecast_pm25["ds"] > last_date_pm25]
    forecast_pm25 = forecast_pm25[["ds", "yhat"]].rename(columns={"yhat": "PM2.5"})

    # PM10
    pm10_df = df[["datetime", "pm10_value"]].rename(columns={"datetime": "ds", "pm10_value": "y"})
    pm10_df["y"] = pm10_df["y"].fillna(pm10_df["y"].mean())
    model_pm10 = Prophet()
    model_pm10.fit(pm10_df)
    future_pm10 = model_pm10.make_future_dataframe(periods=12, freq="6H")
    last_date_pm10 = pm10_df["ds"].max()
    forecast_pm10 = model_pm10.predict(future_pm10)
    forecast_pm10 = forecast_pm10[forecast_pm10["ds"] > last_date_pm10]
    forecast_pm10 = forecast_pm10[["ds", "yhat"]].rename(columns={"yhat": "PM10"})

    # Merge
    merged = (
        forecast_no2
        .merge(forecast_so2, on="ds")
        .merge(forecast_pm25, on="ds")
        .merge(forecast_pm10, on="ds")
    )
    merged.to_csv("pollution.csv", index=False)



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/current")
def current():
    df = pd.read_csv("pol_data.csv")
    latest_record = df.iloc[-1].to_dict()
    aqi = compute_aqi(latest_record)
    return render_template("current.html", record=latest_record, aqi=aqi)

@app.route("/forecast")
@app.route("/forecast")
def forecast():
    generate_forecast()
    df = pd.read_csv("pollution.csv")
    if df.empty:
        return "No forecast data generated."

    # Convert to datetime and extract date
    df["ds"] = pd.to_datetime(df["ds"])
    df["date"] = df["ds"].dt.date

    # Group by day and compute average pollutants
    daily = df.groupby("date").mean().reset_index()
    daily = daily.head(3)  # Only next 3 days

    results = []
    for _, row in daily.iterrows():
        day_data = {
            "date": row["date"].strftime("%Y-%m-%d"),
            "NO2": round(row["NO2"], 2),
            "SO2": round(row["SO2"], 2),
            "PM2.5": round(row["PM2.5"], 2),
            "PM10": round(row["PM10"], 2),
        }

        # Calculate weighted pollution index
        index = (
            row["NO2"] * 0.3 +
            row["SO2"] * 0.2 +
            row["PM2.5"] * 0.25 +
            row["PM10"] * 0.25
        )
        index = round(index, 1)

        # Decide alert level
        if index < 150:
            level = "Safe"
            suggestions = "Normal outdoor activity."
            diseases = "None."
        elif index < 250:
            level = "Moderate"
            suggestions = "Reduce outdoor time. Avoid heavy exercise outside."
            diseases = "Possible mild eye/respiratory irritation."
        elif index < 350:
            level = "High"
            suggestions = "Wear mask outdoors. Avoid jogging or running."
            diseases = "Asthma, coughing, difficulty breathing."
        else:
            level = "Very High"
            suggestions = "Stay indoors. Use N95 mask if stepping out."
            diseases = "Severe respiratory risk. May affect heart health."

        day_data.update({
            "pollution_index": index,
            "alert_level": level,
            "suggestions": suggestions,
            "diseases": diseases
        })
        results.append(day_data)

    return render_template("forecast_cards.html", forecasts=results)

if __name__ == "__main__":
    app.run(debug=True)