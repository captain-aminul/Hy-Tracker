from training_helper import training_the_dataset

if __name__=="__main__":
    best_weight = training_the_dataset()
    print("Best training file is saved:", best_weight)