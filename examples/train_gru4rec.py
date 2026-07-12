from deepsklearn.config import amazon_beauty_config
from deepsklearn.models import  GRU4Rec
from deepsklearn.datasets import TorchStreamingGenerativeDataset
from torch.utils.data import DataLoader
from deepsklearn.trainer import GenerativeTrainer
from deepsklearn.utils import Logger,set_seed,prevent_sleep
import torch.nn as nn
'''
2026-07-11 16:34:32 | INFO | generative_trainer.py:106 | {'model': 'gru4rec', 'duration': '207.405min', 'stage': 'training', 'epoch': 199, 'step_size': 363, 'step_loss': 6.310811996459961, 'ema_loss': 7.028778616422429, 'global_size': 4472600, 'global_step': 4600}
2026-07-11 16:34:44 | INFO | generative_trainer.py:163 | {'stage': 'validation', 'model_name': 'gru4rec', 'epoch': 199, 'validation_number': 22363, 'validation_loss': 7.3063}
2026-07-11 16:34:57 | INFO | generative_trainer.py:163 | {'stage': 'validation', 'model_name': 'gru4rec', 'epoch': 199, 'validation_number': 22363, 'validation_loss': 7.3063}
2026-07-11 16:34:57 | INFO | generative_trainer.py:136 | restore the best model weight to the current model

'''
logger=Logger.get_logger()
set_seed(42)
prevent_sleep() #prevent_sleep

############setu up config##########
generative_train_data_path=amazon_beauty_config.generative_train_data
generative_validation_data_path=amazon_beauty_config.generative_validation_data
generative_feature_columns=amazon_beauty_config.generative_feature_columns
generative_label_column=amazon_beauty_config.generative_label_column
generative_sequence_columns=amazon_beauty_config.generative_sequence_columns
train_batch_size=1000
validation_batch_size=1000
device='cpu'
epoch_number = 200
use_warm_up = True
warm_up_steps = 10
validation_steps = 20
use_early_stop = True
#define model
model_name = "gru4rec"
num_items = amazon_beauty_config.n_items
seq_len = amazon_beauty_config.generative_seq_len
model = GRU4Rec(
    num_items=num_items,
    embed_dim=32,
    seq_len=seq_len)

def main(model:nn.Module):
    train_dataset = TorchStreamingGenerativeDataset(
        data_path=generative_train_data_path,
        feature_columns=generative_feature_columns,
        sequence_columns=generative_sequence_columns,
        label_column=generative_label_column,
        batch_size=train_batch_size
    )
    train_dataloader = DataLoader(train_dataset, batch_size=None)
    validation_dataset = TorchStreamingGenerativeDataset(
        data_path=generative_validation_data_path,
        feature_columns=generative_feature_columns,
        sequence_columns=generative_sequence_columns,
        label_column=generative_label_column,
        batch_size=validation_batch_size
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=None)
    trainer = GenerativeTrainer(
        model_name=model_name,
        model=model,
        device=device,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epoch_number=epoch_number,
        use_warm_up=use_warm_up,
        warm_up_steps=warm_up_steps,
        validation_steps=validation_steps,
        use_early_stop=use_early_stop
    )
    trainer.train()

if __name__ == '__main__':
    main(model)









