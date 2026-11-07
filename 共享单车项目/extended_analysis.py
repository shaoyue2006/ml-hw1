"""Additional descriptive analysis on training data only."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from preprocessing import ROOT, load_and_split

train, _ = load_and_split()
out = ROOT / '图表'
report_dir = ROOT / '数据'
plt.style.use('seaborn-v0_8-whitegrid')

hour = train.groupby(['workingday', 'hr']).cnt.agg(['mean', 'median', 'count']).reset_index()
weather = train.groupby('weathersit').cnt.agg(['mean', 'median', 'count']).reset_index()
month = train.groupby(['yr', 'mnth']).cnt.agg(['mean', 'median', 'count']).reset_index()
for name, df in [('hourly_summary', hour), ('weather_summary', weather), ('monthly_summary', month)]:
    df.to_csv(report_dir / (name + '.csv'), index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(10, 5))
pivot = train.pivot_table(index='weekday', columns='hr', values='cnt', aggfunc='mean')
im = ax.imshow(pivot, aspect='auto', cmap='YlOrRd')
ax.set(title='Average rentals by weekday and hour (training data)', xlabel='Hour', ylabel='Weekday (dataset code)')
ax.set_xticks(range(0, 24, 2)); ax.set_yticks(range(7))
fig.colorbar(im, ax=ax, label='Average rentals'); fig.tight_layout()
fig.savefig(out / '05_weekday_hour_heatmap.png', dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for yr, group in month.groupby('yr'):
    axes[0].plot(group.mnth, group['mean'], marker='o', label=str(2011 + yr))
axes[0].set(title='Monthly averages', xlabel='Month', ylabel='Average hourly rentals'); axes[0].legend()
axes[1].bar(weather.weathersit.astype(str), weather['mean'], color='#2878b5')
for i, row in weather.reset_index(drop=True).iterrows():
    axes[1].text(i, row['mean'], f"n={int(row['count'])}", ha='center', va='bottom', fontsize=8)
axes[1].set(title='Weather categories', xlabel='Weather code', ylabel='Average hourly rentals')
fig.suptitle('Training data descriptive summaries'); fig.tight_layout()
fig.savefig(out / '06_month_weather.png', dpi=160); plt.close(fig)

working = hour[hour.workingday == 1].sort_values('mean', ascending=False)
nonworking = hour[hour.workingday == 0].sort_values('mean', ascending=False)
summary = {'working_peak': working[['hr', 'mean', 'count']].head(3).to_dict('records'),
           'nonworking_peak': nonworking[['hr', 'mean', 'count']].head(3).to_dict('records'),
           'weather': weather.to_dict('records'),
           'correlation': train[['temp','atemp','hum','windspeed','cnt']].corr()['cnt'].round(3).to_dict(),
           'training_distribution': train.cnt.describe(percentiles=[.25,.5,.75,.95]).to_dict(),
           'feature_cardinality': {k: int(train[k].nunique()) for k in ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']}}
(report_dir / 'eda_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
