import datetime
import pickle
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from functions import add_lags, createFeatures


class RecieptCountPredictor(ABC):
    def __init__(self, model_name) -> None:
        with open(f"models/{model_name}.pckl", "rb") as mod:
            try:
                self.model = pickle.load(mod)
            except (OSError, FileNotFoundError, TypeError):
                print("wrong path/ model not available")
                exit(-1)
    @abstractmethod
    def predict(self, target_date) -> List:
        pass

    @abstractmethod
    def preprocess_inputs(self, target_date):
        pass

    @abstractmethod
    def postprocess_outputs(self, output_from_model) -> List:
        pass
    

class FBProphetPredictor(RecieptCountPredictor):
    def __init__(self) -> None:
        super().__init__("fbprophet")
    
    def preprocess_inputs(self, target_date):
        target_date_series = pd.DataFrame({"ds": pd.date_range(start=target_date, end=target_date)})
        return target_date_series
    
    def postprocess_outputs(self, output_from_model) -> List:
        return output_from_model["yhat"].tolist()
    
    def predict(self,target_date)->List:
        pred = self.model.predict(self.preprocess_inputs(target_date))
        pred = self.postprocess_outputs(pred)
        return pred
    

class XGBPredictor(RecieptCountPredictor):
    def __init__(self) -> None:
        super().__init__("xgb")
        self.df = pd.read_csv("data/df_xgb.csv",index_col='# Date') 

    def preprocess_inputs(self, target_date):
        future = pd.date_range(start=target_date, end=target_date)
        future_df = pd.DataFrame(index=future)
        future_df['isFuture'] = True
        df_combined = pd.concat([self.df, future_df])
        df_combined.index=pd.to_datetime(df_combined.index)
        df_combined = createFeatures(df_combined)
        df_combined = add_lags(df_combined)
        
        
        target_date_features = df_combined[df_combined['isFuture'] == True]

        target_date_features = target_date_features.drop(['isFuture', 'Receipt_Count'], axis=1)
        return target_date_features

    def postprocess_outputs(self, output_from_model) -> List:
        last_prediction = float(output_from_model[-1])
        return last_prediction

    def predict(self, target_date) -> List:
        target_date_features = self.preprocess_inputs(target_date)
        FEATURES = ['dayofweek', 'quarter', 'month','year','dayofmonth','dayofyear','weekoftheyear',
                'lag1','lag2','lag3','lag4']
        pred = self.model.predict(target_date_features[FEATURES])
        pred = self.postprocess_outputs(pred)
        return pred
    
    
    
    