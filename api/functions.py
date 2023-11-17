import pandas as pd


def add_lags(df):
    global add_lags
    target_map =df['Receipt_Count'].to_dict()
    df['lag1']=(df.index-pd.Timedelta('91 days')).map(target_map)
    df['lag2']=(df.index-pd.Timedelta('182 days')).map(target_map)
    df['lag3']=(df.index-pd.Timedelta('273 days')).map(target_map)
    df['lag4']=(df.index-pd.Timedelta('364 days')).map(target_map)
    return df

def createFeatures(df):
    global createFeatures
    df = df.copy()
    df['dayofweek']=df.index.day_of_week
    df['quarter']=df.index.quarter
    df['month']=df.index.month
    df['year']=df.index.year
    df['dayofmonth']=df.index.day
    df['dayofyear']=df.index.dayofyear
    df['weekoftheyear']=df.index.isocalendar().week
    return df

