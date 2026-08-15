import pandas as pd
import numpy as np

train = pd.read_csv('train-test.csv')
rate = train['posted_rate']

print('=== TARGET DISTRIBUTION ===')
print('Skewness:', round(rate.skew(), 3))
print('Kurtosis:', round(rate.kurtosis(), 3))
print('Log skewness:', round(np.log1p(rate).skew(), 3))

rpm = rate / train['distance']
print('\nRate per mile skewness:', round(rpm.skew(), 3))

# Equipment-specific patterns
for eq in ['Dry Van', 'Reefer', 'Flatbed']:
    sub = train[train['equipment'] == eq]
    rpm_eq = sub['posted_rate'] / sub['distance']
    print('{}: mean_rpm={:.3f}, std={:.3f}, n={}'.format(eq, rpm_eq.mean(), rpm_eq.std(), len(sub)))

# City-pair analysis
train['route'] = train['pickup'] + '|' + train['delivery']
route_stats = train.groupby('route')['posted_rate'].agg(['mean','std','count'])
print('\nUnique routes:', len(route_stats))
print('Routes with >10 samples:', (route_stats['count']>10).sum())
print('Routes with >50 samples:', (route_stats['count']>50).sum())

# Market index trend
train['date'] = pd.to_datetime(train['date'])
monthly = train.groupby(train['date'].dt.to_period('M')).agg({
    'market_index': 'mean',
    'posted_rate': 'mean',
    'quote_signal': 'mean'
})
print('\nMonthly trends:')
print(monthly.to_string())

# Within-route correlation with market_index
print('\nCorr of rate with market_index within routes:')
def route_corr(g):
    if len(g) > 5:
        return g['posted_rate'].corr(g['market_index'])
    return np.nan
corrs = train.groupby('route').apply(route_corr).dropna()
print('  Mean corr: {:.3f}'.format(corrs.mean()))
print('  Median corr: {:.3f}'.format(corrs.median()))
print('  Positive corr %: {:.1f}%'.format((corrs>0).mean()*100))

# Within-route correlation with quote_signal
print('\nCorr of rate with quote_signal within routes:')
def route_corr_qs(g):
    if len(g) > 5:
        return g['posted_rate'].corr(g['quote_signal'])
    return np.nan
corrs_qs = train.groupby('route').apply(route_corr_qs).dropna()
print('  Mean corr: {:.3f}'.format(corrs_qs.mean()))
print('  Median corr: {:.3f}'.format(corrs_qs.median()))

# Check if distance * quote_signal is a good predictor
pred_simple = train['distance'] * train['quote_signal']
from sklearn.metrics import r2_score
print('\nR2 of distance*quote_signal: {:.4f}'.format(r2_score(rate, pred_simple)))
print('R2 of distance alone: {:.4f}'.format(r2_score(rate, train['distance'] * rate.mean() / train['distance'].mean())))

# Residual analysis: what does distance*quote explain vs what's left?
from sklearn.linear_model import LinearRegression
X_simple = train[['distance', 'quote_signal', 'market_index']].fillna(train[['distance','quote_signal','market_index']].median())
lr = LinearRegression().fit(X_simple, rate)
print('\nLinear R2 (dist+quote+market): {:.4f}'.format(lr.score(X_simple, rate)))

X_more = train[['distance', 'quote_signal', 'market_index', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']].fillna(
    train[['distance', 'quote_signal', 'market_index', 'weight', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']].median())
lr2 = LinearRegression().fit(X_more, rate)
print('Linear R2 (all numeric): {:.4f}'.format(lr2.score(X_more, rate)))
