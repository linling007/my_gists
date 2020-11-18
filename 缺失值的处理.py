import pandas as pd


def missing_processing(df):
    missing_rate = []
    for col in df:
        feature_missing_rate = round(df[col].isnull().sum()/len(df), 4)
        missing_rate.append(feature_missing_rate)
    missing_df = pd.DataFrame(
        {"cols": df.columns.tolist(), "missing_rate": missing_rate})
    sorted_missing = missing_df.sort_values(by='missing_rate')
    return sorted_missing


def process_missing():
    pass
