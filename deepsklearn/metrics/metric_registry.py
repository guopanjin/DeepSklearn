from deepsklearn.metrics import metric_functions
from deepsklearn.config.constants import Constant

_METRICS_REGISTRY={
    Constant.Metrics.precision:metric_functions.precision_score,
    Constant.Metrics.recall:metric_functions.recall,
    Constant.Metrics.f1_score:metric_functions.f1_score,
    Constant.Metrics.auc:metric_functions.auc,
    Constant.Metrics.gauc:metric_functions.gauc,
    Constant.Metrics.recall_at_k:metric_functions.recall_at_k,
    Constant.Metrics.precision_at_k:metric_functions.precision_at_k,
    Constant.Metrics.ndcg_at_k:metric_functions.ndcg_at_k,
    Constant.Metrics.hr_at_k:metric_functions.hr_at_k,
    Constant.Metrics.mrr_at_k:metric_functions.mrr_at_k
}


def get_metrics(metric_name:str,**kwargs):
    if metric_name not in _METRICS_REGISTRY.keys():
        raise ValueError(f"this is no such:{metric_name}, availabel:{sorted(_METRICS_REGISTRY.keys())}")
    function=_METRICS_REGISTRY.get(metric_name)
    return function(**kwargs)

