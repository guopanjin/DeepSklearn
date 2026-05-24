import torch.optim
import torch.nn as nn
from utils import get_device,set_seed
from features import FeaturePipeline
import torch.nn.functional as  F
from models import LR
from datasets import TorchStreamingDataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from utils import Logger
from sklearn.metrics import roc_auc_score
'''
baseline:
2026-05-24 00:46:10 | INFO | test_LR.py:368 | {'stage': 'validation', 'avg_validate_loss': 0.4945, 'validate_size': 4584062, 'validation_auc': 0.7392346397497195}
TODO:
regularization:
scheduler learning rate.
AdamW(weight_decay=1e-5)
'''
logger=Logger.get_logger()
feature_configs={
    "f1": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f2": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f3": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f4": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f5": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f6": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f7": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f8": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f9": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f10": {
        "type": "continuous",
        "args": {
            "scale": 100
        }
    },
    "f11": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f12": {
        "type": "continuous",
        "args": {
            "scale": 100
        }
    },
    "f13": {
        "type": "continuous",
        "args": {
            "scale": 1
        }
    },
    "f14": {
        "type": "catagorical",
        "args": {
            "bucket_size": 3555
        }
    },
    "f15": {
        "type": "catagorical",
        "args": {
            "bucket_size": 1587
        }
    },
    "f16": {
        "type": "catagorical",
        "args": {
            "bucket_size": 793416
        }
    },
    "f17": {
        "type": "catagorical",
        "args": {
            "bucket_size": 310695
        }
    },
    "f18": {
        "type": "catagorical",
        "args": {
            "bucket_size": 777
        }
    },
    "f19": {
        "type": "catagorical",
        "args": {
            "bucket_size": 39
        }
    },
    "f20": {
        "type": "catagorical",
        "args": {
            "bucket_size": 31998
        }
    },
    "f21": {
        "type": "catagorical",
        "args": {
            "bucket_size": 1626
        }
    },
    "f22": {
        "type": "catagorical",
        "args": {
            "bucket_size": 9
        }
    },
    "f23": {
        "type": "catagorical",
        "args": {
            "bucket_size": 84765
        }
    },
    "f24": {
        "type": "catagorical",
        "args": {
            "bucket_size": 13974
        }
    },
    "f25": {
        "type": "catagorical",
        "args": {
            "bucket_size": 667350
        }
    },
    "f26": {
        "type": "catagorical",
        "args": {
            "bucket_size": 9174
        }
    },
    "f27": {
        "type": "catagorical",
        "args": {
            "bucket_size": 78
        }
    },
    "f28": {
        "type": "catagorical",
        "args": {
            "bucket_size": 25734
        }
    },
    "f29": {
        "type": "catagorical",
        "args": {
            "bucket_size": 516963
        }
    },
    "f30": {
        "type": "catagorical",
        "args": {
            "bucket_size": 30
        }
    },
    "f31": {
        "type": "catagorical",
        "args": {
            "bucket_size": 11289
        }
    },
    "f32": {
        "type": "catagorical",
        "args": {
            "bucket_size": 5406
        }
    },
    "f33": {
        "type": "catagorical",
        "args": {
            "bucket_size": 9
        }
    },
    "f34": {
        "type": "catagorical",
        "args": {
            "bucket_size": 600624
        }
    },
    "f35": {
        "type": "catagorical",
        "args": {
            "bucket_size": 42
        }
    },
    "f36": {
        "type": "catagorical",
        "args": {
            "bucket_size": 45
        }
    },
    "f37": {
        "type": "catagorical",
        "args": {
            "bucket_size": 110532
        }
    },
    "f38": {
        "type": "catagorical",
        "args": {
            "bucket_size": 204
        }
    },
    "f39": {
        "type": "catagorical",
        "args": {
            "bucket_size": 83871
        }
    }
}
label_configs=['label']
def main():
    set_seed(42)
    #debug_train.csv experiment_train.csv
    train_data = '../../data/criteo/experiment_train.csv'
    validation_data='../../data/criteo/experiment_validation.csv'
    batch_size = 20000
    validation_batch_size=1000000
    feature_columns=FeaturePipeline(feature_configs).get_feature_columns()
    logger.info(feature_columns)
    train_dataset = TorchStreamingDataset(data_path=train_data,
                                    feature_configs=feature_configs,
                                    label_configs=label_configs,
                                    batch_size=batch_size
                                    )
    train_dataLoader = DataLoader(
        train_dataset,
        batch_size=None
    )
    validation_dataset = TorchStreamingDataset(data_path=validation_data,
                                          feature_configs=feature_configs,
                                          label_configs=label_configs,
                                          batch_size=validation_batch_size
                                          )

    validation_dataLoader = DataLoader(
        validation_dataset,
        batch_size=None
    )

    model=LR(feature_columns)
    loss_fn=nn.BCEWithLogitsLoss()
    learning_rate=0.001
    optimizer=torch.optim.Adam(params=model.parameters(),weight_decay=1e-5,lr=learning_rate)
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
            batch_size=list(feature_dict.values())[0].shape[0]
            global_size += batch_size
            global_batch_number+=1
            # calculate the loss
            batch_loss=loss.detach().cpu().item()
            window_loss+=batch_loss*batch_size
            window_size+=batch_size
            epoch_loss+=batch_loss*batch_size
            epoch_size+=batch_size
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
        batch_size=list(feature_dict.values())[0].shape[0]
        validate_loss+=batch_loss*batch_size
        validate_size+=batch_size
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

