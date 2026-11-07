"""
Part B - Customer Segments and Lapse Prediction
Instamart Customer Intelligence & Support Copilot Capstone
"""

import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Dynamic Path Resolution
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..")) if os.path.basename(CURRENT_DIR) == "src" else CURRENT_DIR

def find_data_file(filename):
    candidates = [
        os.path.join(PROJECT_ROOT, "data", "clean", filename),
        os.path.join(PROJECT_ROOT, "data", "raw", filename),
        os.path.join(PROJECT_ROOT, "data", filename),
        os.path.join(CURRENT_DIR, "data", "clean", filename),
        os.path.join(CURRENT_DIR, "..", "data", "clean", filename),
        filename
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return None

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

orders_path = find_data_file("orders.csv")
customers_path = find_data_file("customers.csv")
products_path = find_data_file("products.csv")

if not orders_path or not customers_path:
    print("Error: Could not locate clean orders.csv or customers.csv.")
    print("Please make sure Part A has run and produced data/clean/orders.csv and data/clean/customers.csv")
    sys.exit(1)

print(f"Loading data from:\n  Orders: {orders_path}\n  Customers: {customers_path}")

orders = pd.read_csv(orders_path)
customers = pd.read_csv(customers_path)
products = pd.read_csv(products_path) if products_path else None

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# =========================================================
# Task 1: RFM Table
# =========================================================
print("\n" + "="*50)
print("TASK 1: BUILD RFM TABLE")
print("="*50)

completed = orders[orders["status"] == "completed"].copy()
snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

print(f"RFM Customers: {len(rfm):,} (Reference: 1,175)")
print(f"Recency: median = {rfm['recency'].median():.1f}, min = {rfm['recency'].min()}, max = {rfm['recency'].max()} (Ref: 19, 0, 180)")
print(f"Frequency: mean = {rfm['frequency'].mean():.2f}, max = {rfm['frequency'].max()} (Ref: 9.46, 33)")
print(f"Monetary: mean = INR {rfm['monetary'].mean():,.2f}, max = INR {rfm['monetary'].max():,.2f} (Ref: 2,552.06, 14,766.55)")

# =========================================================
# Task 2: K-Means Clustering (K=2..8)
# =========================================================
print("\n" + "="*50)
print("TASK 2: K-MEANS CLUSTERING (K=2..8)")
print("="*50)

scaler_rfm = StandardScaler()
rfm_scaled = scaler_rfm.fit_transform(rfm[["recency", "frequency", "monetary"]])

print(f"{'K':<4} {'Inertia':<10} {'Silhouette':<12}")
print("-" * 28)
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    sil = silhouette_score(rfm_scaled, labels)
    print(f"{k:<4} {km.inertia_:<10.1f} {sil:<12.3f}")

# =========================================================
# Task 3: Profile, Name, PCA & Export Segments
# =========================================================
print("\n" + "="*50)
print("TASK 3: PROFILE, NAME AND PROJECT SEGMENTS (K=4)")
print("="*50)

km4 = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["segment_id"] = km4.fit_predict(rfm_scaled)

profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
).reset_index()

highest_mon_seg = profile.loc[profile["mean_monetary"].idxmax(), "segment_id"]

def assign_segment_name(row):
    sid = row["segment_id"]
    if sid == highest_mon_seg:
        return "High-Value Spenders"
    if row["mean_frequency"] > 15:
        return "Loyal Champions"
    if row["mean_recency"] > 50:
        return "Dormant / At-Risk"
    return "Occasional Shoppers"

profile["segment_name"] = profile.apply(assign_segment_name, axis=1)
seg_name_map = dict(zip(profile["segment_id"], profile["segment_name"]))
rfm["segment_name"] = rfm["segment_id"].map(seg_name_map)
rfm["is_premium"] = rfm["segment_id"] == highest_mon_seg

print("Segment Profiles:")
print(profile.to_string(index=False))

pca = PCA(n_components=2, random_state=42)
pca.fit(rfm_scaled)
evr = pca.explained_variance_ratio_
print(f"\nPCA Explained Variance Ratio: [{evr[0]:.3f}, {evr[1]:.3f}], Total: {sum(evr):.3f} (Ref: ~0.634, ~0.218, total ~0.852)")

seg_output_path = os.path.join(OUTPUT_DIR, "customer_segments.csv")
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
    seg_output_path, index=False
)
print(f"Saved: {seg_output_path}")

# =========================================================
# Task 4: Define Lapse Problem (Cutoff 2025-09-01)
# =========================================================
print("\n" + "="*50)
print("TASK 4: DEFINE LAPSE PROBLEM & COHORT")
print("="*50)

cutoff = pd.Timestamp("2025-09-01 00:00:00")
end_eval = pd.Timestamp("2025-10-01 00:00:00")

pre_orders = orders[orders["order_ts"] < cutoff].copy()
pre_completed = pre_orders[pre_orders["status"] == "completed"].copy()

cohort_cids = sorted(pre_completed["customer_id"].unique())
print(f"Cohort size: {len(cohort_cids):,} (Reference: 1,156)")

eval_completed = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] < end_eval) & (orders["status"] == "completed")]
active_in_eval = set(eval_completed["customer_id"].unique())

if products is not None and "category" in products.columns:
    pre_orders_with_cat = pre_orders.merge(products[["product_id", "category"]], on="product_id", how="left")
else:
    pre_orders_with_cat = pre_orders.copy()
    if "category" not in pre_orders_with_cat.columns:
        pre_orders_with_cat["category"] = "Grocery"

features = []
for cid in cohort_cids:
    c_orders = pre_orders[pre_orders["customer_id"] == cid]
    c_comp = pre_completed[pre_completed["customer_id"] == cid]
    
    last_comp_ts = c_comp["order_ts"].max()
    rec = (cutoff - last_comp_ts).days
    freq = len(c_comp)
    mon = c_comp["order_amount"].sum()
    avg_disc = c_comp["discount_pct"].mean()
    
    final_orders = c_orders[c_orders["status"].isin(["completed", "cancelled"])]
    canc_rate = (c_orders["status"] == "cancelled").sum() / len(final_orders) if len(final_orders) > 0 else 0.0
    
    avg_del = c_comp["delivery_minutes"].mean()
    
    c_comp_with_cat = pre_orders_with_cat[(pre_orders_with_cat["customer_id"] == cid) & (pre_orders_with_cat["status"] == "completed")]
    fav_cat = c_comp_with_cat["category"].mode()[0] if len(c_comp_with_cat) > 0 else "Grocery"
    
    features.append({
        "customer_id": cid,
        "recency": rec,
        "frequency": freq,
        "monetary": mon,
        "avg_discount_pct": avg_disc,
        "cancellation_rate": canc_rate,
        "avg_delivery_minutes": avg_del,
        "favourite_category": fav_cat,
        "lapsed": 1 if cid not in active_in_eval else 0
    })

df_feat = pd.DataFrame(features)
df_feat = df_feat.merge(customers[["customer_id", "city", "gender", "age", "signup_date"]], on="customer_id", how="left")
df_feat["tenure_days"] = (cutoff - df_feat["signup_date"]).dt.days
df_feat = df_feat.drop(columns=["signup_date"])

n_lapsed = df_feat["lapsed"].sum()
pct_lapsed = df_feat["lapsed"].mean() * 100
print(f"Lapsed customers: {n_lapsed:,} ({pct_lapsed:.2f}%) (Reference: 443, 38.32%)")

X = df_feat.drop(columns=["customer_id", "lapsed"])
y = df_feat["lapsed"]

X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
    X, y, df_feat["customer_id"], test_size=0.2, stratify=y, random_state=42
)
print(f"Train set: {len(X_train)} rows, Test set: {len(X_test)} rows (Ref: 924, 232)")

y_baseline = np.zeros_like(y_test)
print(f"Baseline (Always predict Not Lapsed):")
print(f"  Accuracy: {accuracy_score(y_test, y_baseline):.1%} (Ref: 61.6%)")
print(f"  Lapsed-class F1: {f1_score(y_test, y_baseline, zero_division=0):.3f} (Ref: 0)")

# =========================================================
# Task 5: Preprocessing Without Leakage
# =========================================================
print("\n" + "="*50)
print("TASK 5: PREPROCESSING PIPELINE (FIT ON TRAIN ONLY)")
print("="*50)

num_cols = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
cat_cols = ["favourite_category", "city", "gender"]

try:
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
except TypeError:
    ohe = OneHotEncoder(drop="first", sparse=False, handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", ohe, cat_cols)
])

# Compare learned mean vs full cohort mean
scaler_check = StandardScaler()
scaler_check.fit(X_train[["frequency"]])
train_freq_mean = scaler_check.mean_[0]
full_freq_mean = df_feat["frequency"].mean()
print(f"Frequency Scaler Learned Mean (Train only): {train_freq_mean:.4f}")
print(f"Frequency Full Cohort Mean:                {full_freq_mean:.4f}")
print("Note: They differ slightly because the scaler was strictly fitted only on the training set to prevent data leakage.")
print(f"Missing values check in feature table: {X.isna().sum().sum()} missing values found.")

# =========================================================
# Task 6: Train, Tune, and Evaluate
# =========================================================
print("\n" + "="*50)
print("TASK 6: MODEL TUNING & EVALUATION")
print("="*50)

# 1. Logistic Regression
pipe_lr = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

grid_lr = GridSearchCV(
    pipe_lr,
    {"clf__C": [0.01, 0.1, 1, 10, 100]},
    cv=5,
    scoring="f1"
)
grid_lr.fit(X_train, y_train)

best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)
lr_f1 = f1_score(y_test, y_pred_lr)
lr_prec = precision_score(y_test, y_pred_lr)
lr_rec = recall_score(y_test, y_pred_lr)

print(f"Logistic Regression: Best C = {grid_lr.best_params_['clf__C']}, Best CV F1 = {grid_lr.best_score_:.3f}")
print(f"  Test Performance -> Precision: {lr_prec:.3f}, Recall: {lr_rec:.3f}, F1: {lr_f1:.3f} (Ref F1: ~0.62)")

# 2. Random Forest
pipe_rf = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

grid_rf = GridSearchCV(
    pipe_rf,
    {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    },
    cv=5,
    scoring="f1"
)
grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf)
rf_rec = recall_score(y_test, y_pred_rf)

print(f"\nRandom Forest: Best params = {grid_rf.best_params_}, Best CV F1 = {grid_rf.best_score_:.3f}")
print(f"  Test Performance -> Precision: {rf_prec:.3f}, Recall: {rf_rec:.3f}, F1: {rf_f1:.3f} (Ref F1: ~0.65)")

print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_rf))

# =========================================================
# Task 7: Class-Balance Experiment
# =========================================================
print("\n" + "="*50)
print("TASK 7: CLASS-BALANCE OVERSAMPLING EXPERIMENT")
print("="*50)

train_df = pd.concat([X_train, y_train], axis=1)
train_lapsed = train_df[train_df["lapsed"] == 1]
train_active = train_df[train_df["lapsed"] == 0]
diff = len(train_active) - len(train_lapsed)

oversampled_lapsed = train_lapsed.sample(n=diff, replace=True, random_state=42)
train_balanced = pd.concat([train_df, oversampled_lapsed], axis=0)

X_train_bal = train_balanced.drop(columns=["lapsed"])
y_train_bal = train_balanced["lapsed"]

print(f"Original Train: Active={len(train_active)}, Lapsed={len(train_lapsed)}")
print(f"Balanced Train: Active={(y_train_bal == 0).sum()}, Lapsed={(y_train_bal == 1).sum()} (Ref: 570 each)")

pipe_lr_bal = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(C=grid_lr.best_params_["clf__C"], max_iter=1000, random_state=42))
])
pipe_lr_bal.fit(X_train_bal, y_train_bal)
y_pred_bal = pipe_lr_bal.predict(X_test)

bal_prec = precision_score(y_test, y_pred_bal)
bal_rec = recall_score(y_test, y_pred_bal)
bal_f1 = f1_score(y_test, y_pred_bal)

print(f"\nUnbalanced LR: Precision={lr_prec:.3f}, Recall={lr_rec:.3f}, F1={lr_f1:.3f}")
print(f"Balanced LR:   Precision={bal_prec:.3f}, Recall={bal_rec:.3f}, F1={bal_f1:.3f}")
print(f"Result: Recall jumped significantly from {lr_rec:.3f} to {bal_rec:.3f} while precision remained near {bal_prec:.3f}.")

# =========================================================
# Task 8: Interpret, Persist Model & Retention List
# =========================================================
print("\n" + "="*50)
print("TASK 8: INTERPRET, PERSIST & RETENTION LIST")
print("="*50)

# Top 5 Random Forest Features
cat_encoder = best_rf.named_steps["prep"].named_transformers_["cat"]
cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(cat_feature_names)

rf_importances = best_rf.named_steps["clf"].feature_importances_
feat_imp = pd.Series(rf_importances, index=all_feature_names).sort_values(ascending=False)
print("Top 5 Features by Random Forest Importance:")
for name, val in feat_imp.head(5).items():
    print(f"  {name:<25}: {val:.4f}")

# Persist Model
model_path = os.path.join(OUTPUT_DIR, "lapse_model.joblib")
joblib.dump(best_rf, model_path)
print(f"\nSaved model to: {model_path}")

# Verify Reload
reloaded_model = joblib.load(model_path)
reloaded_preds = reloaded_model.predict(X_test)
assert np.array_equal(reloaded_preds, y_pred_rf), "Model reload verification failed!"
print("Model reload check: EXACT MATCH with original predictions.")

# Retention List for Premium Segment
test_probs = best_rf.predict_proba(X_test)[:, 1]
test_eval_df = pd.DataFrame({
    "customer_id": test_ids.values,
    "lapse_probability": np.round(test_probs, 4),
    "actually_lapsed": y_test.values
})

test_eval_df = test_eval_df.merge(rfm[["customer_id", "is_premium"]], on="customer_id", how="left")
premium_test_df = test_eval_df[test_eval_df["is_premium"] == True].sort_values("lapse_probability", ascending=False)

retention_top10 = premium_test_df.head(10)[["customer_id", "lapse_probability", "actually_lapsed"]]
retention_path = os.path.join(OUTPUT_DIR, "retention_list.csv")
retention_top10.to_csv(retention_path, index=False)

print(f"\nSaved: {retention_path}")
print("Top 10 Retention List (Premium Customers Ranked by Lapse Risk):")
print(retention_top10.to_string(index=False))
print(f"Actually lapsed count among top 10: {retention_top10['actually_lapsed'].sum()} / 10")

print("\n" + "="*50)
print("PART B PIPELINE COMPLETED SUCCESSFULLY!")
print("="*50)
