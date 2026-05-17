from base import FeatureProcessor
import numpy as np
from pandas import DataFrame
import pandas as pd
from utils.hash_utils import xxhash_encoder
class ContinuousProcessor(FeatureProcessor):

    def __init__(self,scale=1.0,granularity=10,B=150):
        self.scale=scale
        self.granularity=granularity
        self.B=B

    def transformer(self,series:pd.Series):
        if pd.api.types.is_numeric_dtype(series):
            np_values=series.values.astype(float)
            mask=np.isnan(np_values)
            transformed=np.arcsinh(np_values*self.scale)*self.granularity
            transformed=np.floor(transformed)
            transformed=np.clip(transformed,-self.B,self.B)
            transformed=transformed+self.B
            transformed[mask]=self.B*2+1
            return transformed
        else:
            raise ValueError("ContinuousProcessor requires the data to be numeric")
        pass

class CategoricalProcessor(FeatureProcessor):
    def __init__(self,bucket_size=1000):
        self.bucket_size=bucket_size
    def transformer(self,series:pd.Series):
        #fill the missing data
        x=series.fillna("MISSING").astype(str).values
        x=np.array([
            xxhash_encoder(v)%self.bucket_size for v in x
        ])
        return x
if __name__ == '__main__':
    print(np.arcsinh(0))
    print(np.clip(0.1,10,80))
    df=pd.DataFrame({
        "f1":[1,2,3,4,5,None,9]
    })
    print(df)
    print(df.values)
    print(type((df.values)))
    print(np.floor(12.34))
    np_values=df.values
    mask=np.isnan(np_values)
    df1 = pd.DataFrame({
        "f1": ["2323jjjj", 2, 3, 4, 5, None, 9]
    })
    continuousProcessor=ContinuousProcessor()
    #print(continuousProcessor.transformer(df1["f1"]))
    df2 = pd.DataFrame({
        "f1": ["2323jjjj", "erwerw", 3, 4, 5, None, 9]
    })
    print(df2["f1"].values,df2["f1"].values.shape)
    print(df["f1"].values,df["f1"].values.shape)
    categoricalProcessor=CategoricalProcessor()
    x=categoricalProcessor.transformer(df2)
    print(x)
    pass