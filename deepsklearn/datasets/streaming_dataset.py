import os
import pandas as pd
from torch.utils.data import IterableDataset,get_worker_info
from deepsklearn.features import FeaturePipeline
from deepsklearn.utils import Logger
'''
Build the streamingDataset based on the pytorch API
return (feature_dict,label_dict)
'''
logger=Logger.get_logger()
class  TorchStreamingDataset(IterableDataset):
   def __init__(self,data_path,feature_configs,label_configs,batch_size=1000):
       self.data_path=data_path
       self.batch_size=batch_size
       self.feature_configs=feature_configs
       self.feature_pipeline=FeaturePipeline(self.feature_configs)
       self.label_configs=label_configs
       self.file_list=sorted(self.__get_file_list())# make sure the dataset is stable
   def __get_file_list(self):
       file_list=[]
       if os.path.isdir(self.data_path):
           for root, dirs, files in os.walk(self.data_path):
               for file in files:
                   file_list.append(os.path.join(root, file))
       else:
           file_list.append(self.data_path)
       logger.info(f"file_list:{file_list}")
       return file_list

   def __parse_data(self,file):
       for batch in pd.read_csv(file,chunksize=self.batch_size):
           feature_dict=self.feature_pipeline.transform(batch)
           #return 1D array shape=(N,)
           label_dict={label_config:batch[label_config].to_numpy() for label_config in self.label_configs}
           yield (feature_dict,label_dict)
   def __iter__(self):
       worker_info=get_worker_info()
       if worker_info is None:
           for file in self.file_list:
               yield from self.__parse_data(file)
       else:
           worker_number= worker_info.num_workers
           worker_id= worker_info.id
           for index,file in enumerate(self.file_list):
               if index%worker_number==worker_id:
                   yield from self.__parse_data(file)

