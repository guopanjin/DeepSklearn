from deepsklearn.config import criteo_config
from deepsklearn.features import FeaturePipeline
from deepsklearn.utils import Logger,set_seed
from deepsklearn.datasets import TorchStreamingDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from deepsklearn.models import LR
from deepsklearn.trainer import Trainer
from deepsklearn.utils import prevent_sleep

'''
2026-07-02 18:34:28 | INFO | train.py:99 | {'duration': '25.404min', 'stage': 'training', 'epoch': 0, 'step_size': 20000, 'step_loss': 0.4671539068222046, 'step_auc': 0.7844866029034115, 'ema_loss': 0.4656378122810304, 'global_size': 36600000, 'global_step': 1830}
2026-07-02 18:35:55 | INFO | train.py:142 | {'stage': 'validation', 'epoch': 0, 'validation_number': 4584062, 'validation_auc': 0.787538365202911, 'validation_loss': 0.4599}


'''

logger=Logger.get_logger()
set_seed(42)

train_data=criteo_config.train_data
validation_data= criteo_config.validation_data

feature_config=criteo_config.feature_config
label_config=criteo_config.label_config

batch_size=20000
validation_batch_size = 2000000
feature_columns = FeaturePipeline(feature_config).get_feature_columns()
model_name="lr"
model=LR(feature_columns=feature_columns,customize_init_embedding=True)
def main(model_name,model:nn.Module):
    train_dataset = TorchStreamingDataset(
        data_path=train_data,
        feature_configs=feature_config,
        label_configs=label_config,
        batch_size=batch_size
    )
    train_dataLoader = DataLoader(
        train_dataset,
        batch_size=None
    )
    validation_dataset = TorchStreamingDataset(data_path=validation_data,
                                               feature_configs=feature_config,
                                               label_configs=label_config,
                                               batch_size=validation_batch_size
                                               )
    validation_dataLoader = DataLoader(
        validation_dataset,
        batch_size=None
    )
    trainer=Trainer(
            model_name=model_name,
            model=model,
            train_dataloader=train_dataLoader,
            validation_dataloader=validation_dataLoader,
            use_warm_up=True,
            warm_up_steps=100
                    )
    trainer.train()

if __name__ == '__main__':
    prevent_sleep()
    main(model_name=model_name,model=model)