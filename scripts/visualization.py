# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv(r"C:\Users\sunee\flight-price-analyst\data\cleaned_flights_analysis.csv")
print(df.shape)
print(df.head())
# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['Price'], bins=50, kde=True, color='steelblue', ax=axes[0])
axes[0].set_title('Distribution of Flight Prices')
axes[0].set_xlabel('Price (Rs.)')
axes[0].set_ylabel('Number of Flights')

sns.boxplot(x=df['Price'], color='lightcoral', ax=axes[1])
axes[1].set_title('Price Spread & Outliers')
axes[1].set_xlabel('Price (Rs.)')

plt.tight_layout()
plt.savefig(r"C:\Users\sunee\flight-price-analyst\charts\01_price_distribution.png", dpi=150)

print("Skewness:", round(df['Price'].skew(), 3))
print("Median:", round(df['Price'].median(), 2))
print("Mean:", round(df['Price'].mean(), 2))
# %%
# Are expensive flights explained by class/route, or are they noise?
expensive = df[df['Price'] > 300000]
print("Rows above Rs.300,000:", len(expensive), f"({len(expensive)/len(df)*100:.1f}%)")

print("\n--- Travel class breakdown ---")
print(expensive['Travel_Class'].value_counts())
print("\n--- Overall class breakdown for comparison ---")
print(df['Travel_Class'].value_counts())

print("\n--- Median price by class (all data) ---")
print(df.groupby('Travel_Class')['Price'].median().sort_values())

print("\n--- Median price by class + distance bucket ---")
print(df.groupby(['Travel_Class', 'Dist_Bucket'], observed=True)['Price'].median().unstack())
# %%
order = df.groupby('Airline')['Price'].median().sort_values().index

plt.figure(figsize=(13, 7))
sns.boxplot(data=df, x='Price', y='Airline', hue='Airline', order=order,
            palette='viridis', legend=False, showfliers=False)
plt.title('Flight Price Distribution by Airline', fontsize=14)
plt.xlabel('Price (Rs.)')
plt.ylabel('')
plt.tight_layout()
plt.savefig(r"C:\Users\sunee\flight-price-analyst\charts\02_price_by_airline.png", dpi=150)

print(df.groupby('Airline')['Price'].agg(['median', 'mean', 'count']).sort_values('median'))
# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# LEFT: binned lead time, median price
bins = [-1, 3, 7, 14, 21, 30, 45, 60, 90, 120, 400]
labels = ['0-3d', '4-7d', '8-14d', '15-21d', '22-30d', '31-45d', '46-60d', '61-90d', '91-120d', '120d+']
df['Lead_Bin'] = pd.cut(df['Days_Before_Departure'], bins=bins, labels=labels)

binned = df.groupby('Lead_Bin', observed=True)['Price'].median()
axes[0].plot(range(len(binned)), binned.values, marker='o', color='darkorange', linewidth=2)
axes[0].set_xticks(range(len(binned)))
axes[0].set_xticklabels(binned.index, rotation=45)
axes[0].set_title('Median Price by Booking Lead Time')
axes[0].set_xlabel('Days Before Departure')
axes[0].set_ylabel('Median Price (Rs.)')

# RIGHT: Economy only, indexed to its own baseline — isolates the timing effect
eco = df[df['Travel_Class'] == 'Economy']
eco_binned = eco.groupby('Lead_Bin', observed=True)['Price'].median()
baseline = eco_binned.iloc[-1]  # cheapest/earliest bin as reference
axes[1].plot(range(len(eco_binned)), (eco_binned / baseline * 100).values,
             marker='o', color='steelblue', linewidth=2)
axes[1].axhline(100, color='gray', linestyle='--', linewidth=1)
axes[1].set_xticks(range(len(eco_binned)))
axes[1].set_xticklabels(eco_binned.index, rotation=45)
axes[1].set_title('Economy Price Index by Lead Time (earliest booking = 100)')
axes[1].set_xlabel('Days Before Departure')
axes[1].set_ylabel('Price Index')

plt.tight_layout()
plt.savefig(r"C:\Users\sunee\flight-price-analyst\charts\03_booking_timing.png", dpi=150)

print(df.groupby('Lead_Bin', observed=True)['Price'].agg(['median', 'count']))
print("\nEconomy premium for booking 0-3d vs 120d+:",
      f"{(eco_binned.iloc[0] / eco_binned.iloc[-1] - 1) * 100:.1f}%")
# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# LEFT: price vs distance, coloured by class
sample = df.sample(4000, random_state=42)
for cls, color in zip(['Economy','Premium Economy','Business','First'],
                      ['#4C72B0','#55A868','#C44E52','#8172B2']):
    sub = sample[sample['Travel_Class'] == cls]
    axes[0].scatter(sub['Distance_km'], sub['Price'], s=8, alpha=0.4, label=cls, color=color)
axes[0].set_title('Price vs Distance by Travel Class')
axes[0].set_xlabel('Distance (km)')
axes[0].set_ylabel('Price (Rs.)')
axes[0].legend(markerscale=2)

# RIGHT: heatmap of median price
pivot = df.pivot_table(index='Travel_Class', columns='Dist_Bucket',
                       values='Price', aggfunc='median', observed=True)
pivot = pivot.reindex(['Economy','Premium Economy','Business','First'])
pivot = pivot[['<1000km', '1000-3000', '3000-6000', '6000km+']]
sns.heatmap(pivot, annot=True, fmt=',.0f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label':'Median Price (Rs.)'})
axes[1].set_title('Median Price: Class x Distance')
axes[1].set_xlabel('Distance Bucket')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(r"C:\Users\sunee\flight-price-analyst\charts\04_distance_class.png", dpi=150)

print(pivot)
print("\nPrice per km by class:")
print((df.groupby('Travel_Class', observed=True)['Price'].median() /
       df.groupby('Travel_Class', observed=True)['Distance_km'].median()).sort_values())
# %%
# Chart 5: secondary factors (stops, booking channel, season, weekday, aircraft
# type), restricted to Economy — distance and class dominate price so heavily
# (49x range across the heatmap) that smaller effects are invisible when all
# classes are pooled. Holding class fixed makes the secondary effects legible.
fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

for ax, col, title in zip(axes,
        ['Total_Stops', 'Booking_Channel', 'Season'],
        ['Stops', 'Booking Channel', 'Season']):
    eco = df[df['Travel_Class'] == 'Economy']
    med = eco.groupby(col, observed=True)['Price'].median().sort_values()
    ax.bar(range(len(med)), med.values, color='teal', alpha=0.75)
    ax.set_xticks(range(len(med)))
    ax.set_xticklabels(med.index, rotation=30, ha='right')
    ax.set_title(f'Median Economy Price by {title}')
    ax.set_ylabel('Median Price (Rs.)')

plt.tight_layout()
plt.savefig(r"C:\Users\sunee\flight-price-analyst\charts\05_secondary_factors.png", dpi=150)

for col in ['Total_Stops','Booking_Channel','Season','Weekday','Aircraft_Type']:
    med = df[df['Travel_Class']=='Economy'].groupby(col, observed=True)['Price'].median()
    print(f"\n{col}: spread = {(med.max()/med.min()-1)*100:.1f}%")
    print(med.sort_values())