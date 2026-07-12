import os

import numpy as np
import torch
from torch.utils.data import IterableDataset,get_worker_info
from deepsklearn.utils import Logger
import pyarrow.parquet as pq
'''
Build the streamingDataset based on the pytorch API
return (feature_dict,label_dict)
'''
logger=Logger.get_logger()
class  TorchStreamingGenerativeDataset(IterableDataset):
   def __init__(self,data_path,feature_columns,sequence_columns, label_column, batch_size=1000):
       self.data_path=data_path
       self.batch_size=batch_size
       self.feature_columns=feature_columns
       self.label_column=label_column
       self.sequence_columns=sequence_columns
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
       parquet_file = pq.ParquetFile(os.path.expanduser(file))
       for batch in parquet_file.iter_batches(batch_size=self.batch_size):
           batch_df = batch.to_pandas()
           #return 1D array shape=(N,)
           feature_dict={feature_column:torch.tensor(np.stack(batch_df[feature_column].to_numpy(),axis=0)) for feature_column in self.feature_columns}
           label_data=feature_dict[self.label_column][:,1:]
           label_dict={"label":label_data}
           for sequence_column in self.sequence_columns:
            feature_dict[sequence_column]=feature_dict[sequence_column][:,:-1]
           '''
           make sure if the last token of the train sequence is 0,and the loss is also 0
           for example:
           input:[0,0,0,0] label:[2]
           '''
           label_dict["label"]=torch.masked_fill(label_dict["label"],feature_dict[self.label_column]==0,float(0.0))
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

