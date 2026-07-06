from deepsklearn.config import criteo_config
from deepsklearn.features import FeaturePipeline
from deepsklearn.utils import Logger,set_seed,prevent_sleep
from deepsklearn.datasets import TorchStreamingDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from deepsklearn.models import WideDeep
from deepsklearn.trainer import Trainer
'''
2026-07-03 20:55:20 | INFO | train.py:105 | {'model': 'widedeep', 'duration': '22.432min', 'stage': 'training', 'epoch': 0, 'step_size': 20000, 'step_loss': 0.4512960910797119, 'step_auc': 0.8032659670132741, 'ema_loss': 0.44954721752285864, 'global_size': 36600000, 'global_step': 1830}
2026-07-03 20:56:22 | INFO | train.py:151 | {'stage': 'validation', 'model_name': 'widedeep', 'epoch': 0, 'validation_number': 4584062, 'validation_auc': 0.8054611413192503, 'validation_loss': 0.4445}


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
model_name = "widedeep"
device='cpu'
model = WideDeep(feature_columns=feature_columns,
            norm=False,
            dropout=0,
            customize_init_embedding=True)
def main(model_name,model:nn.Module,device):
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
            warm_up_steps=100,
            device=device
                    )
    trainer.train()

if __name__ == '__main__':
    prevent_sleep()
    main(model_name=model_name,model=model,device=device)