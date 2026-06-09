from deepsklearn.features.base import FeatureProcessor
import numpy as np
import pandas as pd
from deepsklearn.utils.hash_utils import xxhash_encoder
class ContinuousProcessor(FeatureProcessor):
    def __init__(self,scale=1.0,granularity=10,B=150):
        self.scale=scale
        self.granularity=granularity
        self.B=B
    '''
    return 1D-array shape=(N,)
    '''
    def transform(self,series:pd.Series):
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
    '''
    return 1D-array shape=(N,)
    '''
    def transform(self,series:pd.Series):
        #fill the missing data
        x=series.fillna("MISSING").astype(str).values
        x=np.array([
            xxhash_encoder(v)%self.bucket_size for v in x
        ])
        return x