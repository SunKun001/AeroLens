# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv(r"C:\Users\sunee\flight-price-analyst\data\cleaned_flights_analysis.csv")

# Drop columns that can't be used as features
# Flight_ID: unique identifier, no predictive value
# Price_Flagged / Price_Outlier: all False here (already filtered) — no variance
# Dist_Bucket: derived from Distance_km, would be redundant
drop_cols = ['Flight_ID', 'Price_Flagged', 'Price_Outlier', 'Dist_Bucket',
             'Departure_Date', 'Departure_Time', 'Arrival_Time']

X = df.drop(columns=drop_cols + ['Price'])
y = df['Price']

print("Feature columns:", X.columns.tolist())
print("Shape:", X.shape)
print("\nCategorical columns:", X.select_dtypes(include='object').columns.tolist())
print("Numeric columns:", X.select_dtypes(include=np.number).columns.tolist())
#just to differentiate between the two scripts
# %%
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train:", X_train.shape, "| Test:", X_test.shape)

# One-hot encode categoricals, i have made no changes to the numeric columns, just passing them through
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ('num', 'passthrough', num_cols)
    ]
)
# Check how wide the encoded feature matrix becomes
sample = preprocessor.fit_transform(X_train)
print("Encoded feature count:", sample.shape[1])
# %%
#first trying to use a linear regression model, will try other models later if this doesn't work well
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {name} ---")
    print(f"MAE  : Rs.{mae:,.2f}")
    print(f"RMSE : Rs.{rmse:,.2f}")
    print(f"R2   : {r2:.4f}")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

results = []
results.append(evaluate("Linear Regression", y_test, lr_pred))
# %%
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# --- Random Forest ---
rf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ))
])

rf_pipeline.fit(X_train, y_train)
results.append(evaluate("Random Forest", y_test, rf_pipeline.predict(X_test)))

# --- XGBoost ---
xgb_pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    ))
])

xgb_pipeline.fit(X_train, y_train)
results.append(evaluate("XGBoost", y_test, xgb_pipeline.predict(X_test)))

# --- Comparison table ---
print("\n" + "="*50)
print(pd.DataFrame(results).to_string(index=False))
# %%
# REJECTED — see methodology_notes.md #5. Retained for documentation.
# Does training on log(Price) improve things?
xgb_log = Pipeline([
    ('prep', preprocessor),
    ('model', XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=8,
                           subsample=0.8, colsample_bytree=0.8,
                           n_jobs=-1, random_state=42))
])

xgb_log.fit(X_train, np.log1p(y_train))
log_pred = np.expm1(xgb_log.predict(X_test))   # convert back to rupees
results.append(evaluate("XGBoost (log target)", y_test, log_pred))

print("\n" + "="*50)
print(pd.DataFrame(results).to_string(index=False))
# %%
import matplotlib.pyplot as plt

# Get feature names out of the fitted preprocessor
prep = xgb_pipeline.named_steps['prep']
feat_names = list(prep.named_transformers_['cat'].get_feature_names_out(cat_cols)) + num_cols

importances = xgb_pipeline.named_steps['model'].feature_importances_
imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

print("--- Top 20 features ---")
print(imp.head(20))

plt.figure(figsize=(9, 8))
imp.head(20).sort_values().plot(kind='barh', color='teal')
plt.title('XGBoost Feature Importance (Top 20)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(r"C:\Users\sunee\flight-price-analyst\charts\06_feature_importance.png", dpi=150)

# Aggregate one-hot columns back to their source feature
def base_feature(name):
    for c in cat_cols:
        if name.startswith(c + '_'):
            return c
    return name

grouped = imp.groupby(base_feature).sum().sort_values(ascending=False)
print("\n--- Importance by original feature ---")
print(grouped)
# %%
from sklearn.inspection import permutation_importance

# Permutation importance: shuffle each feature, measure how much R2 drops
perm = permutation_importance(xgb_pipeline, X_test, y_test,
                              n_repeats=5, random_state=42, n_jobs=-1,
                              scoring='r2')

perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("--- Permutation importance (R2 drop when shuffled) ---")
print(perm_imp)

# Timing effect held within a single class + route segment
seg = df[(df['Travel_Class']=='Economy') & (df['Dist_Bucket']=='<1000km')]
print("\n--- Economy short-haul: median price by lead time ---")
print(seg.groupby('Lead_Bin' if 'Lead_Bin' in seg else
                  pd.cut(seg['Days_Before_Departure'],
                         bins=[-1,3,7,14,30,60,400],
                         labels=['0-3d','4-7d','8-14d','15-30d','31-60d','60d+']),
                  observed=True)['Price'].median())
# %%
test_out = X_test.copy()
test_out['Actual'] = y_test.values
test_out['Predicted'] = xgb_pipeline.predict(X_test)
test_out['Error'] = test_out['Predicted'] - test_out['Actual']
test_out['Pct_Error'] = (test_out['Error'].abs() / test_out['Actual']) * 100

print("--- Error by travel class ---")
print(test_out.groupby('Travel_Class').agg(
    n=('Actual','size'),
    median_price=('Actual','median'),
    MAE=('Error', lambda e: e.abs().mean()),
    median_pct_err=('Pct_Error','median')
).round(1))

print("\n--- Sample predictions ---")
print(test_out[['Airline','Source','Destination','Travel_Class',
                'Distance_km','Days_Before_Departure',
                'Actual','Predicted','Pct_Error']].head(10).round(1).to_string(index=False))
# %%
import joblib

joblib.dump(xgb_pipeline, r"C:\Users\sunee\flight-price-analyst\models\xgb_price_model.pkl")
print("Model saved.")

# Verify it loads and predicts identically
loaded = joblib.load(r"C:\Users\sunee\flight-price-analyst\models\xgb_price_model.pkl")
print("Reload check — identical predictions:",
      np.allclose(loaded.predict(X_test[:100]), xgb_pipeline.predict(X_test[:100])))
