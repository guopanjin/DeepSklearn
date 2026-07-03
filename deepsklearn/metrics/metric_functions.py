import numpy as np
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score

def precision(*,y_true,y_pred):
    return precision_score(y_true,y_pred)

def recall(*,y_true,y_pred):
    return recall_score(y_true,y_pred)

def f1_score(*,y_true,y_pred):
    return f1_score(y_true,y_pred)

def auc(*,y_true,y_pred)->float:
    return roc_auc_score(y_true,y_pred)
'''
[
   {
      "userid":"userid1",
      "prediction":[],
      "label":[]
   }
]
'''
def uauc(user_scores:list[dict])->float:
    aucs=[]
    for user_score in user_scores:
        prediction=np.array(user_score['prediction'])
        label=np.array(user_score['label'])
        #if that is all positive samples or negative samples,skip
        if np.sum(label)==0 or np.sum(label)==label.shape[0]:
            continue
        auc=roc_auc_score(label,prediction)
        aucs.append(auc)
    if not aucs:
        return float('nan')
    return np.round(np.mean(np.array(aucs)),4)

'''
[
   {
      "userid":"userid1",
      "prediction":[],
      "label":[]
   }
]
'''
def gauc(user_scores:list[dict])->float:
    aucs=[]
    counts=[]
    all_count=0
    for user_score in user_scores:
        prediction=np.array(user_score['prediction'])
        label=np.array(user_score['label'])
        #if that is all positive samples or negative samples,skip
        if np.sum(label)==0 or np.sum(label)==label.shape[0]:
            continue
        auc=roc_auc_score(label,prediction)
        all_count+=label.shape[0]
        counts.append(label.shape[0])
        aucs.append(auc)
    if not aucs:
        return float('nan')
    counts=np.array(counts)
    aucs=np.array(aucs)
    ratio=counts/all_count
    return np.round(np.sum(ratio*aucs),4)

'''
[
  {
    "userid":userid,
    "ranked_list":[id1,id2...idk]
    "click_list":[id2,id5....]
  }
]
'''
def ndcg_at_k(user_scores:list[dict],k)->float:
    ndcg_list=[]
    if not user_scores:
        return float("nan")
    for user_score in user_scores:
        ranked_list=user_score["ranked_list"]
        kk=min(k,len(ranked_list))
        ranked_list=ranked_list[0:kk]
        ranked_array = np.array(ranked_list)
        click_array = np.array(user_score["click_list"])
        bool_index=np.isin(ranked_array,click_array)
        hit_index=np.where(bool_index)[0]+1
        hit_number=hit_index.shape[0]
        if hit_number!=0:
            idcg=np.sum(1/np.log2(1+np.arange(1,hit_number+1,1)))
            dcg=np.sum(1/np.log2(1+hit_index))
            ndcg_list.append(dcg/idcg)
        else:
             ndcg_list.append(0)
    return np.round(np.mean(ndcg_list),4)

'''
[
  {
    "userid":userid,
    "ranked_list":[id1,id2...idk]
    "click_list":[id2,id5....]
  }
]
'''
def recall_at_k(user_scores:list[dict],k)->float:
    recall_list=[]
    if not user_scores:
        return float("nan")
    for user_score in user_scores:
        ranked_list=user_score["ranked_list"]
        kk=min(k,len(ranked_list))
        ranked_list=ranked_list[0:kk]
        ranked_array = np.array(ranked_list)
        click_array = np.array(user_score["click_list"])
        bool_index=np.isin(ranked_array,click_array)
        hit_index=np.where(bool_index)[0]+1
        hit_number=hit_index.shape[0]
        click_number=click_array.shape[0]
        if click_number!=0:
            recall_list.append(hit_number/click_number)
        else:
            continue
    return np.round(np.mean(recall_list),4)
'''
[
  {
    "userid":userid,
    "ranked_list":[id1,id2...idk]
    "click_list":[id2,id5....]
  }
]
'''
def precision_at_k(user_scores:list[dict],k)->float:
    precision_list=[]
    if not user_scores:
        return float("nan")
    for user_score in user_scores:
        ranked_list=user_score["ranked_list"]
        kk=min(k,len(ranked_list))
        ranked_list=ranked_list[0:kk]
        ranked_array = np.array(ranked_list)
        click_array = np.array(user_score["click_list"])
        bool_index=np.isin(ranked_array,click_array)
        hit_index=np.where(bool_index)[0]+1
        hit_number=hit_index.shape[0]
        click_number=click_array.shape[0]
        if kk!=0:
            precision_list.append(hit_number/kk)
    return np.round(np.mean(precision_list),4)

def hr_at_k(user_scores:list[dict],k)->float:
    if not user_scores:
        return float("nan")
    all_user_number=0
    all_hit_user_number=0
    for user_score in user_scores:
        ranked_list=user_score["ranked_list"]
        kk=min(k,len(ranked_list))
        ranked_list=ranked_list[0:kk]
        ranked_array = np.array(ranked_list)
        click_array = np.array(user_score["click_list"])
        bool_index=np.isin(ranked_array,click_array)
        hit_index=np.where(bool_index)[0]+1
        hit_number=hit_index.shape[0]
        click_number=click_array.shape[0]
        if click_number==0:
            continue
        all_user_number+=1
        if hit_number>0:
            all_hit_user_number+=1
    if all_user_number==0:
        return float('nan')
    return np.round(all_hit_user_number/all_user_number,4)

def mrr_at_k(user_scores:list[dict],k)->float:
    mrr_list=[]
    if not user_scores:
        return float("nan")
    for user_score in user_scores:
        ranked_list=user_score["ranked_list"]
        kk=min(k,len(ranked_list))
        ranked_list=ranked_list[0:kk]
        ranked_array = np.array(ranked_list)
        click_array = np.array(user_score["click_list"])
        bool_index=np.isin(ranked_array,click_array)
        hit_index=np.where(bool_index)[0]+1
        hit_number=hit_index.shape[0]
        click_number=click_array.shape[0]
        if click_number==0:
            continue
        if hit_number!=0:
            mrr_list.append(1/hit_index[0])
        else:
            mrr_list.append(0)
    return np.round(np.mean(mrr_list),4)