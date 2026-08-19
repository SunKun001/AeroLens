# %%
import pandas as pd
import numpy as np
import re

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(r"C:\Users\sunee\Downloads\flight_pricing_dataset.csv")
print("Raw shape:", df.shape)
print(df.columns.tolist())

# %%
# ============================================================
# 2. INITIAL INSPECTION
# ============================================================
print(df.info())
print("\n--- Missing values (raw) ---")
print(df.isnull().sum())
print("\n--- Duplicate rows ---")
print(df.duplicated().sum())

# %%
# ============================================================
# 3. FORMAT CLEANING (no data removed — only made machine-readable)
# ============================================================
df_clean = df.copy()

# --- 3.1 Price: strip "Rs." and commas, convert to float ---
def clean_price(val):
    if pd.isna(val):
        return None
    val = str(val).replace("Rs.", "").replace(",", "").strip()
    try:
        return float(val)
    except:
        return None

df_clean['Price'] = df_clean['Price'].apply(clean_price)

# --- 3.2 Distance_km: numeric conversion ---
df_clean['Distance_km'] = pd.to_numeric(df_clean['Distance_km'], errors='coerce')

# --- 3.3 Duration: unify 3 formats ("3h 11m", "1.67", "177 min") into minutes ---
def clean_duration(val):
    if pd.isna(val):
        return None
    val = str(val).strip()
    if 'h' in val or ('m' in val and 'min' not in val):
        hours = re.search(r'(\d+)h', val)
        mins = re.search(r'(\d+)m', val)
        h = int(hours.group(1)) if hours else 0
        m = int(mins.group(1)) if mins else 0
        return h * 60 + m
    elif 'min' in val:
        match = re.search(r'(\d+)', val)
        return int(match.group(1)) if match else None
    else:
        try:
            return float(val) * 60  # decimal hours -> minutes
        except:
            return None

df_clean['Duration_mins'] = df_clean['Duration'].apply(clean_duration)
df_clean.drop(columns=['Duration'], inplace=True)

# --- 3.4 Total_Stops: text -> numeric ---
stop_map = {'non-stop': 0, '1 stop': 1, '2 stops': 2}
def clean_stops(val):
    if pd.isna(val):
        return None
    val = str(val).strip().lower()
    if val in stop_map:
        return stop_map[val]
    try:
        return int(val)
    except:
        return None

df_clean['Total_Stops'] = df_clean['Total_Stops'].apply(clean_stops)

# --- 3.5 Days_Before_Departure: numeric ---
df_clean['Days_Before_Departure'] = pd.to_numeric(df_clean['Days_Before_Departure'], errors='coerce')

# --- 3.6 Passenger_Count: number-words -> digits ---
word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6}
def clean_passengers(val):
    if pd.isna(val):
        return None
    val = str(val).strip().lower()
    if val in word_to_num:
        return word_to_num[val]
    try:
        return int(val)
    except:
        return None

df_clean['Passenger_Count'] = df_clean['Passenger_Count'].apply(clean_passengers)

# --- 3.7 Categorical columns: trim whitespace, standardize casing ---
cat_cols = ['Airline', 'Source', 'Destination', 'Travel_Class', 'Season',
            'Weekday', 'Aircraft_Type', 'Booking_Channel']
for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
    df_clean.loc[df_clean[col] == 'Nan', col] = None

# --- 3.8 Source/Destination: unify city name / airport name / IATA code ---
city_map = {
    'Ahmedabad Airport': 'Ahmedabad', 'Amd': 'Ahmedabad',
    'Bangalore Airport': 'Bangalore', 'Blr': 'Bangalore',
    'Bangkok Airport': 'Bangkok', 'Bkk': 'Bangkok',
    'Chennai Airport': 'Chennai', 'Maa': 'Chennai',
    'Delhi Airport': 'Delhi', 'Del': 'Delhi',
    'Doha Airport': 'Doha', 'Doh': 'Doha',
    'Dubai Airport': 'Dubai', 'Dxb': 'Dubai',
    'Frankfurt Airport': 'Frankfurt', 'Fra': 'Frankfurt',
    'Goa Airport': 'Goa', 'Goi': 'Goa',
    'Hyderabad Airport': 'Hyderabad', 'Hyd': 'Hyderabad',
    'Jaipur Airport': 'Jaipur', 'Jai': 'Jaipur',
    'Kolkata Airport': 'Kolkata', 'Ccu': 'Kolkata',
    'London Airport': 'London', 'Lhr': 'London',
    'Mumbai Airport': 'Mumbai', 'Bom': 'Mumbai',
    'New York Airport': 'New York', 'Jfk': 'New York',
    'Pune Airport': 'Pune', 'Pnq': 'Pune',
    'Singapore Airport': 'Singapore', 'Sin': 'Singapore',
    'Sydney Airport': 'Sydney', 'Syd': 'Sydney',
}
df_clean['Source'] = df_clean['Source'].replace(city_map)
df_clean['Destination'] = df_clean['Destination'].replace(city_map)

# --- 3.9 Parse dates ---
df_clean['Departure_Date'] = pd.to_datetime(df_clean['Departure_Date'], errors='coerce')

# --- 3.10 Remove exact duplicate rows ---
before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
print(f"Dropped {before - len(df_clean)} exact duplicate rows")

# %%
# ============================================================
# 4. FLAG SUSPICIOUS PRICES (flag, do NOT delete)
# ============================================================
# Justification: these round values repeat thousands of times, while genuine
# prices appear ~1-3 times each. Strong evidence of placeholder/dummy entries.
placeholder_values = [200000, 2000, 15000, 25000]
df_clean['Price_Flagged'] = df_clean['Price'].isin(placeholder_values)

print("Flagged (suspicious) rows:", df_clean['Price_Flagged'].sum())
print("\n--- Price value counts (top 10) — evidence of placeholders ---")
print(df_clean['Price'].value_counts().head(10))
print("\n--- Price stats: ALL data ---")
print(df_clean['Price'].describe())
print("\n--- Price stats: EXCLUDING flagged ---")
print(df_clean.loc[~df_clean['Price_Flagged'], 'Price'].describe())

# %%
# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================
# Target variable (Price): drop rows — never fabricate the thing you're predicting
df_clean = df_clean.dropna(subset=['Price'])

# Numeric features: median (robust to outliers)
numeric_features = ['Distance_km', 'Duration_mins', 'Days_Before_Departure',
                    'Total_Stops', 'Passenger_Count']
for col in numeric_features:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Categorical features: mode (most frequent)
categorical_features = ['Airline', 'Source', 'Destination', 'Travel_Class', 'Season',
                        'Weekday', 'Aircraft_Type', 'Booking_Channel',
                        'Departure_Time', 'Arrival_Time']
for col in categorical_features:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# Date: median date
df_clean['Departure_Date'] = df_clean['Departure_Date'].fillna(df_clean['Departure_Date'].median())

# %%
# ============================================================
# 5b. FLAG CONTEXTUAL PRICE OUTLIERS (flag, do NOT delete)
# ============================================================
# Justification: a second contamination pattern sits on top of the flat
# placeholder values above — extreme prices injected at random regardless
# of route or class (e.g. Economy Mumbai->Pune hops at Rs.300k+, a 6000km+
# Economy ticket at Rs.156). A flat threshold can't catch this: Rs.150,000
# is normal for long-haul Business and absurd for short-haul Economy. So
# outliers are judged within Travel_Class + distance peer groups instead.
df_clean['Dist_Bucket'] = pd.cut(df_clean['Distance_km'],
                                 bins=[0, 1000, 3000, 6000, 20000],
                                 labels=['<1000km', '1000-3000', '3000-6000', '6000km+'])

def flag_group_outliers(g):
    q1, q3 = g['Price'].quantile([0.25, 0.75])
    iqr = q3 - q1
    return (g['Price'] < q1 - 3*iqr) | (g['Price'] > q3 + 3*iqr)

# 3x IQR (not the usual 1.5x) — deliberately conservative, so legitimate
# long-haul premium fares (a real fat tail) survive while injected noise
# is caught.
# Fit the fences on the placeholder-excluded population only: the placeholder
# spike (200000 x 8,989, plus 2000/15000/25000) sits inside these same
# Travel_Class + Dist_Bucket groups and would otherwise corrupt each group's
# Q1/Q3. Placeholder rows are already excluded from analysis via Price_Flagged,
# so they default to Price_Outlier = False here.
non_placeholder = df_clean.loc[~df_clean['Price_Flagged']]
outlier_flags = (non_placeholder.groupby(['Travel_Class', 'Dist_Bucket'],
                                         observed=True, group_keys=False)
                                 .apply(flag_group_outliers))
df_clean['Price_Outlier'] = outlier_flags.reindex(df_clean.index, fill_value=False)

n_outliers = df_clean['Price_Outlier'].sum()
print(f"Contextual price outliers flagged: {n_outliers} ({n_outliers / len(non_placeholder):.2%})")
print("Price skew (placeholder-excluded, before outlier exclusion):", non_placeholder['Price'].skew())

# %%
# ============================================================
# 6. FINAL VERIFICATION & SAVE
# ============================================================
print("Final shape:", df_clean.shape)
print("Total missing values:", df_clean.isnull().sum().sum())
print("\n--- Final dtypes ---")
print(df_clean.dtypes)

# Analysis-ready subset (excludes flagged placeholders AND contextual outliers)
df_analysis = df_clean.loc[~df_clean['Price_Flagged'] & ~df_clean['Price_Outlier']].copy()
print("\nAnalysis subset shape (flagged + outliers excluded):", df_analysis.shape)
print("Price skew (analysis subset, after excluding outliers/placeholders):", df_analysis['Price'].skew())

# Save both versions
df_clean.to_csv(r"C:\Users\sunee\flight-price-analyst\data\cleaned_flights_full.csv", index=False)
df_analysis.to_csv(r"C:\Users\sunee\flight-price-analyst\data\cleaned_flights_analysis.csv", index=False)
print("\nSaved both datasets.")
# %%
# ============================================================
# 7. PREPARE DATA FOR SUPABASE UPLOAD
# ============================================================
df_db = df_clean.copy()

# Postgres convention: lowercase column names (avoids quoting every column)
df_db.columns = [c.lower() for c in df_db.columns]

# Format date as string for clean CSV/DB insertion
df_db['departure_date'] = pd.to_datetime(df_db['departure_date']).dt.strftime('%Y-%m-%d')

df_db.to_csv(r"C:\Users\sunee\flight-price-analyst\data\flights_for_db.csv", index=False)

print("Columns:", df_db.columns.tolist())
print("Shape:", df_db.shape)
print("Saved flights_for_db.csv")