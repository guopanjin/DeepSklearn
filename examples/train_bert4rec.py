from deepsklearn.config import amazon_beauty_config
from deepsklearn.models import  Bert4Rec
from deepsklearn.datasets import TorchStreamingBertDataset
from torch.utils.data import DataLoader
from deepsklearn.trainer import GenerativeTrainer
from deepsklearn.utils import Logger,set_seed,prevent_sleep
import torch.nn as nn
'''


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
model_name = "bert4rec"
num_items = amazon_beauty_config.n_items
num_classes=num_items+2
seq_len = amazon_beauty_config.generative_seq_len
model = Bert4Rec(
    num_items=num_items,
    embed_dim=32,
    seq_len=seq_len+1,
    num_layers=2,
    num_heads=2)

def main(model:nn.Module):
    train_dataset = TorchStreamingBertDataset(
        mask_id=num_items+1,
        mask_strategy="random",
        data_path=generative_train_data_path,
        feature_columns=generative_feature_columns,
        sequence_columns=generative_sequence_columns,
        label_column=generative_label_column,
        batch_size=train_batch_size
    )
    train_dataloader = DataLoader(train_dataset, batch_size=None)
    validation_dataset = TorchStreamingBertDataset(
        mask_id=num_items + 1,
        mask_strategy="last_mask",
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
        use_early_stop=use_early_stop,
        num_classes=num_classes
    )
    trainer.train()

if __name__ == '__main__':
    main(model)









