from deepsklearn.config import criteo_config
from deepsklearn.features import FeaturePipeline
from deepsklearn.utils import Logger,set_seed,prevent_sleep
from deepsklearn.datasets import TorchStreamingDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from deepsklearn.models import PNN
from deepsklearn.trainer import DiscriminativeTrainer
'''
2026-07-04 15:24:18 | INFO | generative_trainer.py:105 | {'model': 'pnn', 'duration': '26.039min', 'stage': 'training', 'epoch': 0, 'step_size': 20000, 'step_loss': 0.4476205110549927, 'step_auc': 0.806814190025321, 'ema_loss': 0.44494486984297277, 'global_size': 36600000, 'global_step': 1830}
2026-07-04 15:25:57 | INFO | generative_trainer.py:151 | {'stage': 'validation', 'model_name': 'pnn', 'epoch': 0, 'validation_number': 4584062, 'validation_auc': 0.8102533904656681, 'validation_loss': 0.4399}

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
model_name = "pnn"
device='cpu'
model = PNN(feature_columns=feature_columns,
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
    trainer=DiscriminativeTrainer(
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