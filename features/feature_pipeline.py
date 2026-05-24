import pandas as pd
from features.build_feature_processor import BuildProcessor
import torch

class FeaturePipeline:
    def __init__(self,feature_configs:dict):
        self.feature_configs=feature_configs
        self.processor_dict=self.__initialize_process(self.feature_configs)
    def __initialize_process(self,feature_configs:dict):
        processor_dict={}
        for feature_name,cfg in feature_configs.items():
            processor_dict[feature_name]=BuildProcessor.registry_processor(cfg)
        return processor_dict

    # for streaming trainner,so we need to input data here
    def transform(self,batch:pd.DataFrame):
        feature_dict={}
        for feature_name,feature_process in self.processor_dict.items():
            #The pytorch requires the id is long Type
            feature_dict[feature_name]=torch.from_numpy(feature_process.transform(batch[feature_name])).to(torch.long)
        return feature_dict
    def get_feature_columns(self):
        feature_columns={}
        for key,value in self.feature_configs.items():
            type=value['type']
            if type=='continuous':
                feature_columns[key]=302
            else:
                feature_columns[key]=value['args']['bucket_size']
        return feature_columns



