from deepsklearn.config import amazon_beauty_config
from deepsklearn.utils import Logger,set_seed,prevent_sleep
from deepsklearn.datasets import TorchStreamingSeqDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from deepsklearn.models import AvgPooling
from deepsklearn.trainer import DiscriminativeTrainer
'''

dropout=0.2,earystop, epoch_number=200,
 
###avg_pooling:50
2026-07-07 19:45:34 | INFO | generative_trainer.py:179 | {'stage': 'validation', 'model_name': 'avg_pooling', 'epoch': 25, 'validation_number': 44726, 'validation_auc': 0.7243282333671035, 'validation_loss': 0.6342}
2026-07-07 19:45:34 | INFO | generative_trainer.py:211 | early stop,stop training, best_auc:0.7252544271466989,best_loss:0.6295, bad_round:5, min_delta:0.0005
2026-07-07 19:45:34 | INFO | generative_trainer.py:149 | restore the best model weight to the current model
 
###avg_poling real_seq_len
2026-07-07 19:55:11 | INFO | generative_trainer.py:179 | {'stage': 'validation', 'model_name': 'avg_pooling', 'epoch': 19, 'validation_number': 44726, 'validation_auc': 0.7262610782283466, 'validation_loss': 0.665}
2026-07-07 19:55:11 | INFO | generative_trainer.py:211 | early stop,stop training, best_auc:0.7325087775933158,best_loss:0.6394, bad_round:5, min_delta:0.0005
2026-07-07 19:55:11 | INFO | generative_trainer.py:149 | restore the best model weight to the current model


'''
logger=Logger.get_logger()
set_seed(42)

train_data=amazon_beauty_config.train_dataset
validation_data=amazon_beauty_config.validation_dataset

feature_columns=amazon_beauty_config.feature_columns
label_columns=amazon_beauty_config.label_columns

batch_size=10000
validation_batch_size = 10000
logger.info(f"feature_columns:{feature_columns}")
model_name = "avg_pooling"
device='cpu'
num_items=amazon_beauty_config.n_items
model = AvgPooling(num_items=num_items,
            drop_out=0.2
            )
def main(model_name,model:nn.Module,device):
    train_dataset = TorchStreamingSeqDataset(
        data_path=train_data,
        feature_columns=feature_columns,
        label_columns=label_columns,
        batch_size=batch_size
    )
    train_dataLoader = DataLoader(
        train_dataset,
        batch_size=None
    )
    validation_dataset = TorchStreamingSeqDataset(
        data_path=validation_data,
        feature_columns=feature_columns,
        label_columns=label_columns,
        batch_size=batch_size)

    validation_dataLoader = DataLoader(
        validation_dataset,
        batch_size=None
    )
    trainer=DiscriminativeTrainer(
            model_name=model_name,
            model=model,
            train_dataloader=train_dataLoader,
            validation_dataloader=validation_dataLoader,
            epoch_number=200,
            use_warm_up=True,
            warm_up_steps=10,
            validation_steps= 200,
            use_early_stop=True,
            device=device
                    )
    trainer.train()

if __name__ == '__main__':
    prevent_sleep()
    main(model_name=model_name,model=model,device=device)