import pandas as pd
import numpy as np 
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

def load_and_preprocess(csv_path="data/monaco_2024_laps.csv"):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    le.fit(COMPOUND_ORDER)
    df["CompoundEnc"] = le.transform(df["Compound"].str.upper().fillna("MEDIUM"))
    df = df[df["LapTime_s"].between(70, 120)].copy()
    df = df[df["TyreLife"].between(1, 50)].copy()
    return df,le

def build_features(df):
    X = np.column_stack([
        df["CompoundEnc"],
        df["TyreLife"],
        df["TyreLife"] ** 2,
        df["LapNumber"],
        df["PitOut"],
    ])
    y = df["LapTime_s"].values
    return X, y

def train(csv_path="data/monaco_2024_laps.csv"):
    df, le = load_and_preprocess(csv_path)
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"✅ Tyre model MAE: {mae:.3f} seconds")
    return model, le, df

def save_model(model, le, path="models/tyre_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "le": le}, f)
    print(f"💾 Model saved → {path}")


def load_model(path="models/tyre_model.pkl"):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["model"], obj["le"]

def predict_lap_time(model, le, compound, tyre_life, lap_number, pit_out=False):
    comp_enc = le.transform([compound.upper()])[0]
    X = np.array([[comp_enc, tyre_life, tyre_life**2, lap_number, int(pit_out)]])
    return float(model.predict(X)[0])

def plot_degradation_curves(model, le):
    os.makedirs("visualization", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"SOFT": "#E8002D", "MEDIUM": "#FFF200", "HARD": "#CCCCCC"}
    for compound, color in colors.items():
        ages = np.arange(1, 40)
        times = [predict_lap_time(model, le, compound, age, 20) for age in ages]
        ax.plot(ages, times, label=compound, color=color, linewidth=2.5)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.set_xlabel("Tyre Age (laps)", fontsize=12)
    ax.set_ylabel("Predicted Lap Time (s)", fontsize=12)
    ax.set_title("Monaco 2024 – Tyre Degradation Curves", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#2a2a4e", labelcolor="white", fontsize=11)
    ax.grid(True, alpha=0.2, color="gray")
    plt.tight_layout()
    plt.savefig("visualization/tyre_deg_curves.png", dpi=150, bbox_inches="tight")
    print("📊 Degradation curves saved → visualization/tyre_deg_curves.png")

if __name__ == "__main__":
    model, le, df = train()
    save_model(model, le)
    plot_degradation_curves(model, le)
    print("\n🔍 Sample predictions:")
    for c in ["SOFT", "MEDIUM", "HARD"]:
        t = predict_lap_time(model, le, c, 15, 30)
        print(f"  {c} age=15, lap=30 → {t:.2f}s")
    
