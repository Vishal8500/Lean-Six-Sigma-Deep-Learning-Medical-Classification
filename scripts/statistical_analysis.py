# =====================================================
# FULL STATISTICAL ANALYSIS SCRIPT
# (ANOVA + Regression for DOE Results)
# =====================================================

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ==============================
# 1️⃣ LOAD DATA
# ==============================

# Make sure this path matches your project structure
df = pd.read_csv("logs/results.csv")

print("\nTotal rows in dataset:", len(df))

# Keep only DOE runs
df_doe = df[df["run_type"] == "doe"].copy()

print("Total DOE runs:", len(df_doe))

# ==============================
# 2️⃣ PREPARE DATA
# ==============================

# Convert to categorical
df_doe["lr"] = df_doe["lr"].astype(str)
df_doe["batch_size"] = df_doe["batch_size"].astype(str)
df_doe["dropout"] = df_doe["dropout"].astype(str)
df_doe["optimizer"] = df_doe["optimizer"].astype(str)

# ==============================
# 3️⃣ ANOVA ANALYSIS
# ==============================

print("\n========== ANOVA ANALYSIS ==========\n")

anova_model = smf.ols(
    "accuracy ~ C(lr) + C(batch_size) + C(dropout) + C(optimizer)",
    data=df_doe
).fit()

anova_table = sm.stats.anova_lm(anova_model, typ=2)

print(anova_table)

# ==============================
# 4️⃣ REGRESSION ANALYSIS
# ==============================

print("\n========== REGRESSION ANALYSIS ==========\n")

reg_model = smf.ols(
    "accuracy ~ C(lr) + C(batch_size) + C(dropout) + C(optimizer)",
    data=df_doe
).fit()

print(reg_model.summary())

# ==============================
# 5️⃣ FACTOR MEAN EFFECTS
# ==============================

print("\n========== MEAN ACCURACY BY FACTOR ==========\n")

print("Learning Rate Impact:")
print(df_doe.groupby("lr")["accuracy"].mean(), "\n")

print("Dropout Impact:")
print(df_doe.groupby("dropout")["accuracy"].mean(), "\n")

print("Optimizer Impact:")
print(df_doe.groupby("optimizer")["accuracy"].mean(), "\n")

print("Batch Size Impact:")
print(df_doe.groupby("batch_size")["accuracy"].mean(), "\n")

print("Statistical Analysis Completed Successfully.")