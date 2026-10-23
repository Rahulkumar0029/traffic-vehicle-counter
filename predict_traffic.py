import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Predict future traffic trends and analyze traffic flow rates.")
    parser.add_argument("--timeseries", type=Path, default=Path("output/traffic_prediction_timeseries.csv"))
    parser.add_argument("--events", type=Path, default=Path("output/events.csv"))
    parser.add_argument("--output", type=Path, default=Path("output/traffic_forecast.json"))
    parser.add_argument("--forecast-windows", type=int, default=5, help="Number of 10s windows to forecast")
    return parser.parse_args()


def predict_traffic():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    timeseries_path = args.timeseries if args.timeseries.is_absolute() else base_dir / args.timeseries
    events_path = args.events if args.events.is_absolute() else base_dir / args.events
    output_path = args.output if args.output.is_absolute() else base_dir / args.output

    if not timeseries_path.exists():
        print(f"Error: {timeseries_path} does not exist. Run vehicle.py first to generate data.")
        return

    windows = []
    with open(timeseries_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            windows.append({
                "time_window_s": float(row["time_window_s"]),
                "outbound": int(row["outbound"]),
                "inbound": int(row["inbound"]),
                "total": int(row["total"]),
            })

    if not windows:
        print("No traffic window data found to forecast.")
        return

    totals = [w["total"] for w in windows]
    avg_per_window = sum(totals) / len(totals)
    
    # Calculate simple moving trend (slope)
    if len(totals) >= 2:
        trend = (totals[-1] - totals[0]) / (len(totals) - 1)
    else:
        trend = 0.0

    # Hourly flow projection
    # Each window is 10 seconds -> 360 windows per hour
    vehicles_per_hour = avg_per_window * 360

    if vehicles_per_hour < 600:
        congestion_level = "Low / Free Flow"
    elif vehicles_per_hour < 1500:
        congestion_level = "Moderate"
    else:
        congestion_level = "Heavy / Congested"

    # Future forecast for next N windows
    last_time = windows[-1]["time_window_s"]
    forecasts = []
    current_val = float(totals[-1])
    for i in range(1, args.forecast_windows + 1):
        future_time = last_time + (i * 10)
        predicted_count = max(0, round(current_val + (trend * i), 1))
        forecasts.append({
            "future_time_s": future_time,
            "predicted_vehicle_count": predicted_count,
        })

    forecast_result = {
        "historical_windows_analyzed": len(windows),
        "average_vehicles_per_10s": round(avg_per_window, 2),
        "projected_hourly_rate": round(vehicles_per_hour, 1),
        "congestion_status": congestion_level,
        "traffic_trend_per_window": round(trend, 2),
        "next_forecast_windows": forecasts,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(forecast_result, f, indent=2)

    print("\n" + "=" * 50)
    print(" TRAFFIC PREDICTION & FORECAST REPORT")
    print("=" * 50)
    print(f"Windows Analyzed     : {len(windows)}")
    print(f"Avg Vehicles / 10s   : {avg_per_window:.2f}")
    print(f"Projected Hourly Rate: {vehicles_per_hour:.1f} vehicles/hour")
    print(f"Congestion Status    : {congestion_level}")
    print(f"Trend Direction      : {'Increasing' if trend > 0 else 'Decreasing' if trend < 0 else 'Stable'} ({trend:+.2f}/window)")
    print(f"Forecasts Saved To   : {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    predict_traffic()
