import torch.optim
import torch.nn as nn
from deepsklearn.utils import get_device,set_seed
from deepsklearn.features import FeaturePipeline
import torch.nn.functional as  F
from deepsklearn.models import LR
from deepsklearn.datasets import TorchStreamingDataset
import numpy as np
from torch.utils.data import DataLoader
from deepsklearn.utils import Logger
from sklearn.metrics import roc_auc_score
from deepsklearn.config import criteo_config
'''
baseline:
optimizer=torch.optim.Adam(params=model.parameters(),lr=learning_rate)
2026-05-24 00:46:10 | INFO | test_LR.py:368 | {'stage': 'validation', 'avg_validate_loss': 0.4945, 'validate_size': 4584062, 'validation_auc': 0.7392346397497195}
add regularization:
optimizer=torch.optim.Adam(params=model.parameters(),weight_decay=1e-5,lr=learning_rate)
2026-05-24 16:57:31 | INFO | test_LR.py:376 | {'stage': 'validation', 'avg_validate_loss': 0.4945, 'validate_size': 4584062, 'validation_auc': 0.7392334486026022}
change the optimizer from adam to adamw.
optimizer=torch.optim.AdamW(params=model.parameters(),weight_decay=1e-5,lr=learning_rate)
2026-05-24 17:49:51 | INFO | test_LR.py:382 | {'stage': 'validation', 'avg_validate_loss': 0.4945, 'validate_size': 4584062, 'validation_auc': 0.7392346397497195}

## fix the bug of ModuleDict
2026-06-30 18:06:32 | INFO | test_LR.py:152 | {'stage': 'validation', 'avg_validate_loss': 0.46, 'validate_size': 4584062, 'validation_auc': 0.7875293362189391}

debug data:
2026-06-30 19:37:57 | INFO | test_LR.py:150 | {'stage': 'validation', 'avg_validate_loss': 0.5102, 'validate_size': 100000, 'validation_auc': 0.7123905832997076}
'''
logger=Logger.get_logger()
set_seed(42)
feature_config=criteo_config.feature_config
label_config=criteo_config.label_config
train_data=criteo_config.debug_train_data
validation_data=criteo_config.debug_validation_data
batch_size=20000
validation_batch_size = 1000000
def main():
    feature_columns=FeaturePipeline(feature_config).get_feature_columns()
    logger.info(feature_columns)
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

    model=LR(feature_columns)
    loss_fn=nn.BCEWithLogitsLoss()
    learning_rate=0.001
    optimizer=torch.optim.AdamW(params=model.parameters(),weight_decay=1e-5,lr=learning_rate)
    device=get_device()
    logger.info(f"device:{device}")
    model.train()# training model mode
    epoch_number=1
    global_size=0
    global_batch_number=0
    log_batch_interval=10
    log_validation_interval=500
    window_loss=0
    window_size=0
    ema_loss=None
    epoch_loss = 0
    epoch_size=0
    alpha=0.1
    for epoch in range(epoch_number):
        for feature_dict, label_dict in train_dataLoader:
            optimizer.zero_grad()
            logits=model(feature_dict)
            label=torch.unsqueeze(label_dict['label'].to(torch.float32),dim=1)
            loss=loss_fn(logits,label)
            loss.backward()
            optimizer.step()
            cur_batch_size=list(feature_dict.values())[0].shape[0]
            global_size += cur_batch_size
            global_batch_number+=1
            # calculate the loss
            batch_loss=loss.detach().cpu().item()
            window_loss+=batch_loss*cur_batch_size
            window_size+=cur_batch_size
            epoch_loss+=batch_loss*cur_batch_size
            epoch_size+=cur_batch_size
            if(ema_loss==None):
                ema_loss=batch_loss
            else:
                ema_loss=alpha*batch_loss+(1-alpha)*ema_loss
            step_auc = roc_auc_score(label.detach().cpu().numpy(), F.sigmoid(logits).detach().cpu().numpy())
            if global_batch_number % log_validation_interval ==0 and global_batch_number>0:
                validation_metrics(validation_dataLoader, model, loss_fn)
            if global_batch_number%log_batch_interval==0 and global_batch_number>0:
                avg_window_loss=np.round(window_loss/window_size,4)
                logger.info({
                    "stage":"train",
                    "epoch":epoch,
                    "batch_loss":batch_loss,
                    "batch_size": batch_size,
                    "avg_window_loss":avg_window_loss,
                    "window_size": window_size,
                    "ema_loss": ema_loss,
                    "step_auc": step_auc,
                    "global_size":global_size,
                    "global_batch_number":global_batch_number
                })
                window_loss = 0
                window_size = 0
        #one epoch finished
        avg_epoch_loss=np.round(epoch_loss/epoch_size,4)
        logger.info({
            "epoch":epoch,
            "avg_epoch_loss":avg_epoch_loss,
            "epoch_size":epoch_size
        })
        epoch_loss = 0
        epoch_size = 0
        validation_metrics(validation_dataLoader,model,loss_fn)

@torch.no_grad()
def validation_metrics(validation_loader,model,loss_fn):
    model.eval() # evaluation mode
    predict_list=[]
    label_list=[]
    validate_loss=0
    validate_size=0
    for feature_dict, label_dict in validation_loader:
        logits=model(feature_dict)
        label = torch.unsqueeze(label_dict['label'].to(torch.float32), dim=1)
        loss=loss_fn(logits,label)
        label_list.extend(list(label.detach().cpu().numpy().squeeze(axis=1)))
        predict_list.extend(list(F.sigmoid(logits).detach().cpu().numpy().squeeze(axis=1)))
        batch_loss = loss.detach().cpu().item()
        cur_batch_size=list(feature_dict.values())[0].shape[0]
        validate_loss+=batch_loss*cur_batch_size
        validate_size+=cur_batch_size
    avg_validate_loss=np.round(validate_loss/validate_size,4)
    auc=roc_auc_score(label_list,predict_list)
    logger.info({
        "stage":"validation",
        "avg_validate_loss":avg_validate_loss,
        "validate_size":validate_size,
        "validation_auc":auc
    })
    model.train()

if __name__ == '__main__':
    main()
    pass

