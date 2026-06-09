from torch.utils.data import DataLoader
from deepsklearn.datasets import *
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

label_configs=["label"]


if __name__ == '__main__':
    criteo_debug_train_csv_path = '../../data/criteo/debug_train.csv'
    batch_size=10

    dataset=TorchStreamingDataset(data_path=criteo_debug_train_csv_path,
                                  feature_configs=feature_configs,
                                  label_configs=label_configs,
                                  batch_size=batch_size
                               )

    i=0
    #dataLoader=DataLoader(dataset,batch_size=None)
    dataLoader=DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=True
    )
    for feature_dict,label_dict in dataLoader:
        i+=1
        print(feature_dict)
        print(feature_dict["f1"].shape)
        if(i>10):
            break;




