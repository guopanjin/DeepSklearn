from deepsklearn.utils.log_utils import Logger
from deepsklearn.utils import get_device
from deepsklearn.optimizers import build_adamw_with_decay_groups
from deepsklearn.optimizers import get_linear_scheduler
from deepsklearn.metrics import auc
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

logger = Logger.get_logger()
import time

'''
1:train model
2:save model
3:load model
4:predict model

train_dataloader:

(
   {
      "f1":tensor[batch_size] 
      "f2":tensor[batch_size]
   },
   {
     "label1":tensor[batch_size] 
   }
)
'''


class DiscriminativeTrainer:
    def __init__(self,
                 *,
                 model_name: str,
                 model: nn.Module,
                 loss_fn=None,
                 device=None,
                 train_dataloader,
                 validation_dataloader,
                 optimizer: torch.optim.Optimizer = None,
                 model_dir=None,
                 epoch_number=1,
                 use_warm_up=False,
                 warm_up_steps=100,
                 validation_steps=500,
                 log_steps=10,
                 ema_loss_alpha=0.1,
                 customize_initialization=False,
                 use_early_stop=False,
                 step_early_stop=False
                 ):
        self.model_name = model_name
        self.model = model
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCEWithLogitsLoss()
        self.optimizer = optimizer if optimizer is not None else build_adamw_with_decay_groups(self.model)
        self.use_warm_up = use_warm_up
        self.warm_up_steps = warm_up_steps
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.model_dir = model_dir
        self.epoch_number = epoch_number
        self.device = device if device is not None else get_device()
        self.validation_steps = validation_steps
        self.log_steps = log_steps
        self.ema_loss_alpha = ema_loss_alpha
        self.customize_initialization = customize_initialization
        self.use_early_stop = use_early_stop
        self.step_early_stop = step_early_stop

    def train(self):
        logger.info(f"device:{self.device}")
        logger.info(f"{self.model_name} structure:\n {self.model}")
        scheduler = None
        if self.use_warm_up:
            scheduler = get_linear_scheduler(optimizer=self.optimizer, warmup_steps=self.warm_up_steps)
        early_stop = None
        if self.use_early_stop:
            early_stop = EarlyStop()
        model = self.model.to(self.device)
        model.train()
        # the merics we need to monitor
        step_size = 0
        step_auc = 0
        step_loss = 0
        global_size = 0
        global_step = 0
        ema_loss = None
        start_time = time.time()
        for epoch in range(self.epoch_number):
            if self.use_early_stop:
                if early_stop.stopped():
                    logger.info(f"early stop trigged,epoch {epoch}")
                    break;
            for feature_dict, label_dict in self.train_dataloader:
                feature_dict = {k: v.to(self.device) for k, v in feature_dict.items()}
                label_dict = {k: v.to(self.device) for k, v in label_dict.items()}
                label = label_dict["label"]
                logits = torch.squeeze(model(feature_dict), -1)  # (batch_size,1)
                loss = self.loss_fn(logits, label.to(torch.float32))
                self.optimizer.zero_grad()  # clear gradient
                loss.backward()  # get the gradient
                self.optimizer.step()
                ####evaluation part
                step_size = label.shape[0]
                global_size += step_size
                global_step += 1
                step_loss = loss.detach().cpu().item()
                predictions = F.sigmoid(logits).detach().cpu().tolist()
                label_list = label.detach().cpu().tolist()
                step_auc = auc(y_true=label_list, y_pred=predictions)
                if ema_loss is None:
                    ema_loss = step_loss
                else:
                    ema_loss = self.ema_loss_alpha * step_loss + (1 - self.ema_loss_alpha) * ema_loss
                if global_step % self.log_steps == 0:
                    end_time = time.time()
                    logger.info({
                        "model": self.model_name,
                        "duration": str(np.round((end_time - start_time) / 60, 3)) + "min",
                        "stage": "training",
                        "epoch": epoch,
                        "step_size": step_size,
                        "step_loss": step_loss,
                        "step_auc": step_auc,
                        "ema_loss": ema_loss,
                        "global_size": global_size,
                        "global_step": global_step
                    })
                if global_step % self.validation_steps == 0:
                    auc_metric, validation_loss = self._evaluation(epoch=epoch, model_name=self.model_name)
                    if self.use_early_stop and self.step_early_stop:
                        if early_stop.step(validation_loss=validation_loss,
                                           validation_auc=auc_metric,
                                           model=self.model
                                           ):
                            break;
                if scheduler:
                    scheduler.step()
            auc_metric, validation_loss = self._evaluation(epoch=epoch, model_name=self.model_name)
            if self.use_early_stop and not self.step_early_stop:
                if early_stop.step(validation_loss=validation_loss,
                                   validation_auc=auc_metric,
                                   model=self.model
                                   ):
                    break;

        if self.use_early_stop and early_stop.best_auc != 0 and early_stop.best_state != None:
            # restore the best model to self.model
            self.model.load_state_dict(early_stop.best_state)  # in place operation
            logger.info("restore the best model weight to the current model")

    def save(self):
        if self.model_dir is None:
            self.model_dir = os.path.expanduser("~/.deepsklearn/models/")
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, self.model_name)
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"successfully save model to the path {model_path}")

    @torch.no_grad()
    def _evaluation(self, *, epoch, model_name):
        self.model.eval()
        loss_sum = 0
        size_sum = 0
        labels = []
        predictions = []
        for feature_dict, label_dict in self.validation_dataloader:
            feature_dict = {k: v.to(self.device) for k, v in feature_dict.items()}
            label_dict = {k: v.to(self.device) for k, v in label_dict.items()}
            label = label_dict["label"]
            logits = torch.squeeze(self.model(feature_dict), dim=-1)
            loss = self.loss_fn(logits, label.to(torch.float32))
            loss_sum += loss.cpu().item() * label.shape[0]
            size_sum += label.shape[0]
            label_list = label.cpu().tolist()
            labels.extend(label_list)
            prediction_list = F.sigmoid(logits).cpu().tolist()
            predictions.extend(prediction_list)
        auc_metric = auc(y_true=labels, y_pred=predictions)
        validation_loss = np.round(loss_sum / size_sum, 4)
        logger.info({
            "stage": "validation",
            "model_name": model_name,
            "epoch": epoch,
            "validation_number": size_sum,
            "validation_auc": auc_metric,
            "validation_loss": validation_loss
        })
        self.model.train()
        return auc_metric, validation_loss


class EarlyStop:
    def __init__(self,
                 patience: int = 5,
                 min_delta=0.0005
                 ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_state = None
        self.best_auc = 0
        self.bad_round = 0
        self.best_loss = 0
        self.is_stop = False

    def step(self, validation_auc, validation_loss, model: nn.Module):
        if validation_auc > self.best_auc + self.min_delta:
            self.best_auc = validation_auc
            self.best_loss = validation_loss
            self.bad_round = 0
            # becase model.state_dict() is reference not copy,so still changing,so need to copy that
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_round += 1
        if self.bad_round >= self.patience:
            logger.info(
                f"early stop,stop training, best_auc:{self.best_auc},best_loss:{self.best_loss}, bad_round:{self.bad_round}, min_delta:{self.min_delta}")
            self.is_stop = True
            return True
        return False

    def stopped(self):
        return self.is_stop


class Predictor:
    @classmethod
    def load_model(cls, model_path, model: nn.Module, device=None):
        if device is None:
            device = get_device()
        logger.info(f"device={device}")
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model

    '''
    the data needs to be a  dict
    input:
    {
      "feature_name":tensor
    }
    return [0.1,0.2...]
    '''

    @classmethod
    def predict(cls, model: nn.Module, data_dict: dict[str, torch.Tensor], device=None):
        if device is None:
            device = get_device()
        model.eval()
        model = model.to(device)
        data_dict = {k: v.to(device) for k, v in data_dict.items()}
        with torch.no_grad():
            results = F.sigmoid(model(data_dict)).cpu().tolist()
        return results
