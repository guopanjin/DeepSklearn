from deepsklearn.metrics import get_metrics
from deepsklearn.config.constants import Constant


if __name__ == '__main__':
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.6, 0.7, 0.4]

    parameter={"y_true":y_true,
               "y_pred":y_score}
    auc=get_metrics(Constant.Metrics.auc,**parameter)
    print(auc)