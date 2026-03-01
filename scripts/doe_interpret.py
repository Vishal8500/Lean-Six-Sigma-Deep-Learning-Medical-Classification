# =============================
# DOE Interpretation Script
# =============================

import pandas as pd

# Load your results file (update path if needed)
file_path = r"D:\LEAN PROJ\logs\results.csv.xlsx"   # change if running locally
df = pd.read_excel(file_path)

# Separate baseline and DOE
baseline_df = df[df["run_type"] == "baseline"]
doe_df = df[df["run_type"] == "doe"]

# -----------------------------
# 1️⃣ Summary
# -----------------------------

total_runs = len(df)
baseline_accuracy = baseline_df["accuracy"].values[0]
baseline_dpmo = baseline_df["dpmo"].values[0]

best_doe_row = doe_df.sort_values(by="accuracy", ascending=False).iloc[0]
worst_doe_row = doe_df.sort_values(by="accuracy", ascending=True).iloc[0]

best_accuracy = best_doe_row["accuracy"]
worst_accuracy = worst_doe_row["accuracy"]
best_dpmo = best_doe_row["dpmo"]

print("===== DOE SUMMARY =====")
print("Total runs:", total_runs)
print("Baseline Accuracy:", round(baseline_accuracy*100, 2), "%")
print("Best DOE Accuracy:", round(best_accuracy*100, 2), "%")
print("Worst DOE Accuracy:", round(worst_accuracy*100, 2), "%")
print()

# -----------------------------
# 2️⃣ Top 3 Configurations
# -----------------------------

top3 = doe_df.sort_values(by="accuracy", ascending=False).head(3)

print("===== TOP 3 CONFIGURATIONS =====")
print(top3[["lr","batch_size","dropout","optimizer","accuracy","dpmo"]])
print()

# -----------------------------
# 3️⃣ Worst 2 Configurations
# -----------------------------

worst2 = doe_df.sort_values(by="accuracy", ascending=True).head(2)

print("===== WORST CONFIGURATIONS =====")
print(worst2[["lr","batch_size","dropout","optimizer","accuracy"]])
print()

# -----------------------------
# 4️⃣ Improvement Calculation
# -----------------------------

defect_reduction = baseline_dpmo - best_dpmo
percent_reduction = (defect_reduction / baseline_dpmo) * 100

print("===== IMPROVEMENT ANALYSIS =====")
print("Baseline DPMO:", round(baseline_dpmo, 2))
print("Best DOE DPMO:", round(best_dpmo, 2))
print("Defect Reduction:", round(defect_reduction, 2))
print("Percentage Reduction:", round(percent_reduction, 2), "%")
print()

# -----------------------------
# 5️⃣ Factor Insight Analysis
# -----------------------------

print("===== FACTOR MEANS (Average Accuracy by Factor) =====")

print("\nLearning Rate Impact:")
print(doe_df.groupby("lr")["accuracy"].mean().sort_values(ascending=False))

print("\nDropout Impact:")
print(doe_df.groupby("dropout")["accuracy"].mean().sort_values(ascending=False))

print("\nOptimizer Impact:")
print(doe_df.groupby("optimizer")["accuracy"].mean().sort_values(ascending=False))

print("\nBatch Size Impact:")
print(doe_df.groupby("batch_size")["accuracy"].mean().sort_values(ascending=False))

print("\nDOE Interpretation Completed Successfully.")