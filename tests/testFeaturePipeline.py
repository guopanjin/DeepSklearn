from deepsklearn.features import *
import pandas as pd
features_config={
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

def test1():
    criteo_debug_train_csv_path = '../../data/criteo/debug_train.csv'
    feature_pipeline = FeaturePipeline(features_config)
    batch_size = 100
    i = 0;
    for batch in pd.read_csv(criteo_debug_train_csv_path, chunksize=batch_size):
        features_dict = feature_pipeline.transform(batch)
        i += 1
        print(features_dict)
        print("========")
        if i > 2:
            break;
def test2():
    feature_pipeline = FeaturePipeline(features_config)
    feature_columns=feature_pipeline.get_feature_columns()
    print(feature_columns)



if __name__ == '__main__':
    test2()
    pass



