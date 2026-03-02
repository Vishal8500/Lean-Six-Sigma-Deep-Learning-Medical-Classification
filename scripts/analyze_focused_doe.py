import pandas as pd

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("logs/results.csv")

baseline_df = df[df["run_type"] == "baseline"]
doe_df = df[df["run_type"] == "doe"]
focused_df = df[df["run_type"] == "focused_doe"]

print("\nTotal Focused DOE Runs:", len(focused_df))

# ==============================
# BEST CONFIGURATION
# ==============================

best_focused = focused_df.sort_values(by="accuracy", ascending=False).iloc[0]

print("\n===== BEST FOCUSED CONFIG =====")
print(best_focused[[
    "lr", "batch_size", "dropout",
    "optimizer", "accuracy", "dpmo"
]])

# ==============================
# WORST CONFIGURATION
# ==============================

worst_focused = focused_df.sort_values(by="accuracy", ascending=True).iloc[0]

print("\n===== WORST FOCUSED CONFIG =====")
print(worst_focused[[
    "lr", "batch_size", "dropout",
    "optimizer", "accuracy", "dpmo"
]])

# ==============================
# BASELINE COMPARISON
# ==============================

baseline_acc = baseline_df["accuracy"].values[0]
baseline_dpmo = baseline_df["dpmo"].values[0]

best_acc = best_focused["accuracy"]
best_dpmo = best_focused["dpmo"]

print("\n===== BASELINE COMPARISON =====")
print("Baseline Accuracy:", round(baseline_acc, 4))
print("Best Focused Accuracy:", round(best_acc, 4))
print("Accuracy Improvement:",
      round((best_acc - baseline_acc) * 100, 2), "%")

print("\nBaseline DPMO:", round(baseline_dpmo, 2))
print("Best Focused DPMO:", round(best_dpmo, 2))

reduction = baseline_dpmo - best_dpmo
percent = (reduction / baseline_dpmo) * 100

print("Defect Reduction:", round(reduction, 2))
print("Reduction %:", round(percent, 2), "%")

# ==============================
# EFFECT OF LEARNING RATE
# ==============================

print("\n===== MEAN ACCURACY BY LR =====")
print(focused_df.groupby("lr")["accuracy"].mean())

print("\n===== MEAN DPMO BY LR =====")
print(focused_df.groupby("lr")["dpmo"].mean())

# ==============================
# EFFECT OF WEIGHT DECAY
# ==============================

print("\n===== MEAN ACCURACY BY WEIGHT DECAY =====")
print(focused_df.groupby("model_path").count())  # optional debugging

print("\nMean Accuracy by Weight Decay:")
print(focused_df.groupby(focused_df["run_id"].str.extract(r'_(0|0\.0001)$')[0])["accuracy"].mean())

# Better way if weight_decay column exists (if you log it separately):
# print(focused_df.groupby("weight_decay")["accuracy"].mean())

# ==============================
# INTERACTION TABLE
# ==============================

print("\n===== LR vs Weight Decay Interaction (Accuracy) =====")

# Extract weight decay from run_id (since not stored separately)
focused_df["weight_decay"] = focused_df["run_id"].str.extract(r'_(0|0\.0001)$')

pivot_table = focused_df.pivot_table(
    values="accuracy",
    index="lr",
    columns="weight_decay",
    aggfunc="mean"
)

print(pivot_table)

print("\nFocused DOE Analysis Completed Successfully.")