from deepsklearn.config import amazon_beauty_config
from deepsklearn.utils import Logger,set_seed,prevent_sleep
from deepsklearn.datasets import TorchStreamingSeqDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from deepsklearn.models import BST
from deepsklearn.trainer import DiscriminativeTrainer
'''
no dropout,no earystop, epoch_number=200,
2026-07-07 16:49:16 | INFO | generative_trainer.py:105 | {'model': 'bst', 'duration': '59.704min', 'stage': 'training', 'epoch': 199, 'step_size': 2826, 'step_loss': 1.7471880710218102e-05, 'step_auc': 1.0, 'ema_loss': 0.00011824234243329561, 'global_size': 52565200, 'global_step': 5400}
2026-07-07 16:49:17 | INFO | generative_trainer.py:151 | {'stage': 'validation', 'model_name': 'bst', 'epoch': 199, 'validation_number': 44726, 'validation_auc': 0.6933868478803646, 'validation_loss': 6.8213}
2026-07-07 16:49:18 | INFO | generative_trainer.py:151 | {'stage': 'validation', 'model_name': 'bst', 'epoch': 199, 'validation_number': 44726, 'validation_auc': 0.6933868478803646, 'validation_loss': 6.8213}

dropout=0.2,earystop, epoch_number=200,

2026-07-07 17:34:13 | INFO | generative_trainer.py:116 | {'model': 'bst', 'duration': '2.989min', 'stage': 'training', 'epoch': 9, 'step_size': 2826, 'step_loss': 0.234773650765419, 'step_auc': 0.9754919564512923, 'ema_loss': 0.37083536476154827, 'global_size': 2628260, 'global_step': 270}
2026-07-07 17:34:14 | INFO | generative_trainer.py:179 | {'stage': 'validation', 'model_name': 'bst', 'epoch': 9, 'validation_number': 44726, 'validation_auc': 0.7038657781041437, 'validation_loss': 0.8201}
2026-07-07 17:34:14 | INFO | generative_trainer.py:211 | early stop,stop training, best_auc:0.7191011191919252,best_loss:0.6505, bad_round:5, min_delta:0.0005
2026-07-07 17:34:14 | INFO | generative_trainer.py:149 | restore the best model weight to the current model

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
model_name = "bst"
device='cpu'
num_items=amazon_beauty_config.n_items
model = BST(num_items=num_items,
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