# %%
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load(r"C:\Users\sunee\flight-price-analyst\models\xgb_price_model.pkl")
df = pd.read_csv(r"C:\Users\sunee\flight-price-analyst\data\cleaned_flights_analysis.csv")

FEATURES = ['Airline','Source','Destination','Total_Stops','Distance_km',
            'Travel_Class','Days_Before_Departure','Season','Weekday',
            'Aircraft_Type','Booking_Channel','Passenger_Count','Duration_mins']

def forecast_curve(flight: dict, max_days=120):
    """Predict price for one flight across every booking lead time."""
    rows = []
    for d in range(0, max_days + 1):
        f = flight.copy()
        f['Days_Before_Departure'] = d
        rows.append(f)
    grid = pd.DataFrame(rows)[FEATURES]
    return pd.DataFrame({
        'Days_Before_Departure': range(0, max_days + 1),
        'Predicted_Price': model.predict(grid)
    })

# Superseded by forecast_route() — see methodology_notes.md #6. Retained to
# demonstrate why averaging was necessary.
# Taking a real flight from the data as a template
example = df[(df['Source']=='Delhi') & (df['Destination']=='Mumbai') &
             (df['Travel_Class']=='Economy')].iloc[0]
flight = {c: example[c] for c in FEATURES}

curve_single = forecast_curve(flight)
print(f"Route: {flight['Source']} -> {flight['Destination']} "
      f"({flight['Travel_Class']}, {flight['Airline']})")
print(curve_single[curve_single['Days_Before_Departure'].isin([0,3,7,14,21,30,45,60,90,120])].round(0).to_string(index=False))

best = curve_single.loc[curve_single['Predicted_Price'].idxmin()]
worst = curve_single.loc[curve_single['Predicted_Price'].idxmax()]
print(f"\nCheapest at {int(best['Days_Before_Departure'])} days: Rs.{best['Predicted_Price']:,.0f}")
print(f"Priciest at {int(worst['Days_Before_Departure'])} days: Rs.{worst['Predicted_Price']:,.0f}")
print(f"Potential saving: {(worst['Predicted_Price']/best['Predicted_Price']-1)*100:.1f}%")
# %%
def forecast_route(source, dest, travel_class, n_samples=200, max_days=120):
    """Average the forecast curve over many real flights on a route."""
    pool = df[(df['Source']==source) & (df['Destination']==dest) &
              (df['Travel_Class']==travel_class)]
    if len(pool) == 0:
        raise ValueError("No flights match that route/class")
    pool = pool.sample(min(n_samples, len(pool)), random_state=42)

    curves = []
    for _, row in pool.iterrows():
        f = {c: row[c] for c in FEATURES}
        curves.append(forecast_curve(f, max_days)['Predicted_Price'].values)

    avg = np.mean(curves, axis=0)
    return pd.DataFrame({'Days_Before_Departure': range(0, max_days+1),
                         'Predicted_Price': avg})

curve = forecast_route('Delhi', 'Mumbai', 'Economy')

# Smooth residual jitter with a rolling mean
curve['Smoothed'] = curve['Predicted_Price'].rolling(7, center=True, min_periods=1).mean()

checkpoints = [0,3,7,14,21,30,45,60,90,120]
print(curve[curve['Days_Before_Departure'].isin(checkpoints)].round(0).to_string(index=False))

# Compare booking windows rather than single days
early = curve.loc[curve['Days_Before_Departure'].between(60,120), 'Smoothed'].mean()
late  = curve.loc[curve['Days_Before_Departure'].between(0,3),   'Smoothed'].mean()
print(f"\nAvg price booking 60-120 days out: Rs.{early:,.0f}")
print(f"Avg price booking 0-3 days out:    Rs.{late:,.0f}")
print(f"Last-minute premium: {(late/early-1)*100:.1f}%")

# Where does the curve stop improving?
base = curve.loc[curve['Days_Before_Departure']>=90, 'Smoothed'].mean()
within5 = curve[curve['Smoothed'] <= base*1.05]['Days_Before_Departure']
print(f"Prices reach within 5% of floor by {within5.min()} days before departure")
# %%
routes = [('Delhi','Mumbai','Economy'), ('Mumbai','Bangalore','Economy'),
          ('Delhi','Dubai','Economy'), ('Mumbai','London','Economy'),
          ('Delhi','Mumbai','Business'), ('Mumbai','London','Business')]

rows = []
for src, dst, cls in routes:
    try:
        c = forecast_route(src, dst, cls, n_samples=100)
    except ValueError:
        print(f"skipped {src}->{dst} ({cls}) — no matching flights")
        continue
    c['Smoothed'] = c['Predicted_Price'].rolling(7, center=True, min_periods=1).mean()
    early = c.loc[c['Days_Before_Departure'].between(60,120), 'Smoothed'].mean()
    late  = c.loc[c['Days_Before_Departure'].between(0,3),   'Smoothed'].mean()
    base  = c.loc[c['Days_Before_Departure']>=90, 'Smoothed'].mean()
    thresh = c[c['Smoothed'] <= base*1.05]['Days_Before_Departure'].min()
    rows.append({'Route': f"{src}->{dst}", 'Class': cls,
                 'Early': round(early), 'Late': round(late),
                 'Premium_%': round((late/early-1)*100, 1),
                 'Book_By_Days': thresh})

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for src, dst, cls in [('Mumbai','Bangalore','Economy'), ('Delhi','Dubai','Economy'),
                      ('Mumbai','London','Economy')]:
    c = forecast_route(src, dst, cls, n_samples=100)
    c['S'] = c['Predicted_Price'].rolling(7, center=True, min_periods=1).mean()
    base = c.loc[c['Days_Before_Departure']>=90,'S'].mean()
    axes[0].plot(c['Days_Before_Departure'], c['S']/base*100, label=f"{src}->{dst}", linewidth=2)

axes[0].axvline(33, color='crimson', linestyle='--', linewidth=1.2, label='33-day threshold')
axes[0].axhline(105, color='gray', linestyle=':', linewidth=1)
axes[0].invert_xaxis()
axes[0].set_title('Economy Price Index by Lead Time (floor = 100)')
axes[0].set_xlabel('Days Before Departure'); axes[0].set_ylabel('Price Index')
axes[0].legend()

for src, dst, cls in [('Delhi','Mumbai','Economy'), ('Delhi','Mumbai','Business'),
                      ('Mumbai','London','Business')]:
    c = forecast_route(src, dst, cls, n_samples=100)
    c['S'] = c['Predicted_Price'].rolling(7, center=True, min_periods=1).mean()
    base = c.loc[c['Days_Before_Departure']>=90,'S'].mean()
    axes[1].plot(c['Days_Before_Departure'], c['S']/base*100, label=f"{src}->{dst} {cls}", linewidth=2)

axes[1].invert_xaxis()
axes[1].set_title('Cabin Class Comparison (floor = 100)')
axes[1].set_xlabel('Days Before Departure'); axes[1].set_ylabel('Price Index')
axes[1].legend()

plt.tight_layout()
plt.savefig(r"C:\Users\sunee\flight-price-analyst\charts\07_forecast_curves.png", dpi=150)
print("Saved forecast chart.")