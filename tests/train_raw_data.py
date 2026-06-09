import random

random.seed(42)
'''
total_number=45840617
'''
def process_criteo_data():
    train_file= "../../data/criteo/train.txt"
    debug_train_file= "../../data/criteo/debug_train.txt"
    debug_validation_file = "../../data/criteo/debug_validation.txt"
    debug_test_file = "../../data/criteo/debug_test.txt"
    experiment_train_file= "../../data/criteo/experiment_train.txt"
    experiment_validation_file = "../../data/criteo/experiment_validation.txt"
    experiment_test_file = "../../data/criteo/experiment_test.txt"
    debug_number = 1_000_000
    total_number=0
    with open(train_file) as f:
        for _ in f:
            total_number+=1

    with open(train_file) as fin, \
         open(debug_train_file,'w') as f_debug_train, \
         open(debug_validation_file,'w') as f_debug_validation, \
         open(debug_test_file,'w') as f_debug_test:
        for i,line in enumerate(fin):
            if i<debug_number*0.8:
                f_debug_train.write(line)
            elif i<debug_number*0.9:
                f_debug_validation.write(line)
            else:
                f_debug_test.write(line)
            if i>=debug_number:
                break;
        print(f"debug train data finished {i}")
    with open(train_file) as fin, \
         open(experiment_train_file,'w') as f_experiment_train, \
         open(experiment_validation_file,'w') as f_experiment_validation, \
         open(experiment_test_file,'w') as f_experiment_test:
        for i,line in enumerate(fin):
            if i<total_number*0.8:
                f_experiment_train.write(line)
            elif i<total_number*0.9:
                f_experiment_validation.write(line)
            else:
                f_experiment_test.write(line)

process_criteo_data()

