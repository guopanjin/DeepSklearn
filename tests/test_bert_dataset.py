from deepsklearn.config import amazon_beauty_config
from deepsklearn.datasets import TorchStreamingBertDataset
from torch.utils.data import DataLoader

generative_train_data=amazon_beauty_config.generative_train_data
generative_validation_data=amazon_beauty_config.generative_validation_data

generative_feature_columns=amazon_beauty_config.generative_feature_columns
generative_label_column=amazon_beauty_config.generative_label_column
generative_sequence_columns=amazon_beauty_config.generative_sequence_columns
n_items=amazon_beauty_config.n_items


train_dataset=TorchStreamingBertDataset(
             mask_strategy="random",
             mask_id=n_items+1,
             data_path=generative_train_data,
             feature_columns=generative_feature_columns,
             sequence_columns=generative_sequence_columns,
             label_column=generative_label_column,
             batch_size=3)

train_dataloader=DataLoader(train_dataset,batch_size=None)
i=0
for features,labels in train_dataset:
    i+=1
    if i>2:
        break;
    print("=====features====")
    print(features)
    for k,v in features.items():
        print(k,v.shape)
    print("====labels====")
    print(labels)
    for k,v in labels.items():
        print(k,v.shape)

