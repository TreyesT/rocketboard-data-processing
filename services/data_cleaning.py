import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_missing_values(data, required_fields=None):
    df = pd.DataFrame(data)

    if required_fields:
        df = df[required_fields]

    # Identify rows with missing values
    missing_vals = df.isna()
    rows_with_missing_values = df[missing_vals.any(axis=1)]

    results = []
    for idx, row in rows_with_missing_values.iterrows():
        missing_fields = missing_vals.columns[missing_vals.loc[idx]].tolist()
        results.append({
            'entry': row.to_dict(),
            'missing_fields': missing_fields
        })

    return results

def outlier_detection(data, required_fields=None, contamination=0.01, n_estimators=100, random_state=None):
    df = pd.DataFrame(data)

    if required_fields:
        df = df[required_fields]

    numeric_cols = df.select_dtypes(include=['number']).columns

    if numeric_cols.empty:
        return {
            'outliers': [],
            'labels': [],
            'num_outliers': 0,
            'message': 'No numeric columns found for outlier detection.'
        }

    # df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df_numeric = df[numeric_cols].dropna()

    # Keep track of which rows were dropped
    dropped_indices = set(df.index) - set(df_numeric.index)

    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    iso_forest.fit(df_numeric)

    predictions = iso_forest.predict(df_numeric)

    # Prepare output structure
    outliers = []
    for row_idx, pred_label in zip(df_numeric.index, predictions):
        if pred_label == -1:  # outlier
            outliers.append({
                'index': row_idx,
                'row_data': df.loc[row_idx].to_dict()
            })

    return {
        'outliers': outliers,
        'num_outliers': len(outliers)
    }