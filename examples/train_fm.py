import os
from deepsklearn.config import criteo_config
from deepsklearn.features import FeaturePipeline
from deepsklearn.utils import Logger,set_seed
from deepsklearn.datasets import TorchStreamingDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from deepsklearn.models import FM
from deepsklearn.trainer import DiscriminativeTrainer
from deepsklearn.utils import prevent_sleep

'''
2026-07-03 17:33:53 | INFO | generative_trainer.py:104 | {'model': 'fm', 'duration': '13.754min', 'stage': 'training', 'epoch': 0, 'step_size': 20000, 'step_loss': 0.45706331729888916, 'step_auc': 0.7960340318371839, 'ema_loss': 0.4547540705483931, 'global_size': 36600000, 'global_step': 1830}
2026-07-03 17:34:50 | INFO | generative_trainer.py:150 | {'stage': 'validation', 'epoch': 0, 'validation_number': 4584062, 'validation_auc': 0.7996743437582361, 'validation_loss': 0.4501}

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
model_name="fm"
model=FM(feature_columns=feature_columns,
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
    trainer=DiscriminativeTrainer(
            model_name=model_name,
            model=model,
            train_dataloader=train_dataLoader,
            validation_dataloader=validation_dataLoader,
            use_warm_up=True,
            warm_up_steps=100,
            device="cpu"
                    )
    trainer.train()

if __name__ == '__main__':
    prevent_sleep()
    main(model_name=model_name,model=model)