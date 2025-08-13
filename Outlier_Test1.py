import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import date

# Read the Excel file into a DataFrame
df = pd.read_csv("D:\\Sample_data_outlier.csv")

print(df)

# -------------------------
#  Univariate Outlier Function
# -------------------------
def find_outlier_flags(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return series.apply(lambda x: 1 if (x < lower_bound or x > upper_bound) else 0)

# -------------------------
#  Apply Univariate Outlier Flags
# -------------------------
df['is_long_duration'] = find_outlier_flags(df['sessionduration'])
df['is_high_talk_time'] = find_outlier_flags(df['totalagenttalkduration'])
df['is_high_hold_time'] = find_outlier_flags(df['totalagentholdduration'])
df['is_high_wait_time'] = find_outlier_flags(df['totalagentalertduration'])
df['is_high_transfer_count'] = find_outlier_flags(df['consultcount'])
df['is_high_ivr_time'] = find_outlier_flags(df['sessionduration'])
df['is_high_acw_time'] = find_outlier_flags(df['totalagentwrapupduration'])
df['is_sentiment_outlier'] = find_outlier_flags(df['Sent_int'])


# -------------------------
#  Multivariate Outlier Detection (Isolation Forest)
# -------------------------
features = [
    'sessionduration',
    'totalagenttalkduration',
    'totalagentholdduration',
    'totalagentalertduration',
    'consultcount',
    'sessionduration',
    'totalagentwrapupduration',
    'Sent_int'
]

iso = IsolationForest(contamination=0.05, random_state=42)
df['is_multivariate_anomaly'] = iso.fit_predict(df[features])
df['is_multivariate_anomaly'] = df['is_multivariate_anomaly'].apply(lambda x: 1 if x == -1 else 0)


# Save the DataFrame to a CSV file
output_file_path = "D:\\Processed_data_outlier.csv"
df.to_csv(output_file_path, index=False)

print(f"DataFrame saved to {output_file_path}")
