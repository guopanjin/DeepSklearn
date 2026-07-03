from deepsklearn.config import criteo_config
from deepsklearn.features import FeaturePipeline
from deepsklearn.utils import Logger,set_seed,prevent_sleep
from deepsklearn.datasets import TorchStreamingDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from deepsklearn.models import DNN
from deepsklearn.trainer import Trainer
'''
basic:
2026-07-02 19:30:44 | INFO | train.py:99 | {'model': 'dnn', 'duration': '60.515min', 'stage': 'training', 'epoch': 0, 'step_size': 20000, 'step_loss': 0.45818954706192017, 'step_auc': 0.7951053910426619, 'ema_loss': 0.457087082865604, 'global_size': 36600000, 'global_step': 1830}
2026-07-02 19:31:57 | INFO | train.py:143 | {'stage': 'validation', 'epoch': 0, 'validation_number': 4584062, 'validation_auc': 0.7969904831455228, 'validation_loss': 0.4521}

norm=True,
dropout=0.1,
customize_init_embedding=True
2026-07-03 07:58:06 | INFO | train.py:151 | {'stage': 'validation', 'epoch': 0, 'validation_number': 4584062, 'validation_auc': 0.8058751984965479, 'validation_loss': 0.4439}

norm=True,
dropout=0.1,
customize_init_embedding=False
2026-07-03 08:37:01 | INFO | train.py:105 | {'model': 'dnn', 'duration': '24.573min', 'stage': 'training', 'epoch': 0, 'step_size': 20000, 'step_loss': 0.462909996509552, 'step_auc': 0.7901971443948891, 'ema_loss': 0.46177941387995525, 'global_size': 36600000, 'global_step': 1830}
2026-07-03 08:38:11 | INFO | train.py:151 | {'stage': 'validation', 'epoch': 0, 'validation_number': 4584062, 'validation_auc': 0.7953770298259975, 'validation_loss': 0.4531}


norm=False,
dropout=0,
customize_init_embedding=True
2026-07-03 09:47:04 | INFO | train.py:105 | {'model': 'dnn', 'duration': '24.788min', 'stage': 'training', 'epoch': 0, 'step_size': 20000, 'step_loss': 0.4517657160758972, 'step_auc': 0.8020336760364111, 'ema_loss': 0.4501568763407472, 'global_size': 36600000, 'global_step': 1830}
2026-07-03 09:48:14 | INFO | train.py:151 | {'stage': 'validation', 'epoch': 0, 'validation_number': 4584062, 'validation_auc': 0.8051325325380692, 'validation_loss': 0.4448}
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
logger.info(f"feature_columns:{feature_columns}")
model_name = "dnn"
model = DNN(feature_columns=feature_columns,
            norm=False,
            dropout=0,
            customize_init_embedding=True)
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