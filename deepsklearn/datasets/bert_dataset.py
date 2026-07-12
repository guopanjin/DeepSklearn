import os

import numpy as np
import torch
from torch.utils.data import IterableDataset,get_worker_info
from deepsklearn.utils import Logger
import pyarrow.parquet as pq
'''
Build the streamingDataset based on the pytorch API
input:[batch_size,seq_len]

return (feature_dict,label_dict)
mask_strategy:random or last_mask
'''
logger=Logger.get_logger()
class  TorchStreamingBertDataset(IterableDataset):
   def __init__(self,
                *,
                mask_strategy:str,
                mask_id:int,
                data_path,
                feature_columns,
                sequence_columns,
                label_column,
                batch_size=1000,
                mask_ratio:float=0.2
                ):
       self.data_path=data_path
       self.batch_size=batch_size
       self.feature_columns=feature_columns
       self.label_column=label_column
       self.sequence_columns=sequence_columns
       self.mask_strategy=mask_strategy
       self.mask_ratio=mask_ratio
       self.mask_id=mask_id
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
           item_data=feature_dict[self.label_column].clone()
           label_data=feature_dict[self.label_column].clone()
           label_dict={}
           if self.mask_strategy=='random':
               batch_size,seq_len=item_data.shape
               # in case ,sample_number is 0
               sample_number=max(1, int(self.mask_ratio*seq_len))
               random_matrix=torch.rand(batch_size,seq_len)
               cols=torch.argsort(random_matrix,dim=1)[:,0:sample_number]
               rows=torch.unsqueeze(torch.arange(batch_size),dim=-1).expand((batch_size,sample_number))
               item_data[rows,cols]=self.mask_id
               final_label=torch.zeros_like(item_data)
               final_label[rows,cols]=label_data[rows,cols]
               label_dict={"label":final_label.to(torch.long)}
               feature_dict[self.label_column]=item_data
           elif self.mask_strategy=='last_mask':
               item_data[:,-1]=self.mask_id
               final_label=torch.zeros_like(item_data)
               final_label[:, -1]=label_data[:, -1]
               label_dict = {"label": final_label.to(torch.long)}
               feature_dict[self.label_column] = item_data
           else:
               raise ValueError(f"unkown mask strategy ")
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