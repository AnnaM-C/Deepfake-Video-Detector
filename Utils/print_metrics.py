from sklearn.metrics import precision_score, recall_score, f1_score
import csv

def calculate_precision(csv_file):
    predictions = []
    ground_truths = []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            _, prediction, ground_truth = row
            predictions.append(int(prediction))
            ground_truths.append(int(ground_truth))
    precision = precision_score(ground_truths, predictions, zero_division=0)
    return precision

def calculate_recall(csv_file):
    predictions = []
    ground_truths = []    
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            _, prediction, ground_truth = row
            predictions.append(int(prediction))
            ground_truths.append(int(ground_truth))    
    recall = recall_score(ground_truths, predictions, zero_division=0)
    return recall


def calculate_f1(csv_file):
    predictions = []
    ground_truths = []    
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            _, prediction, ground_truth = row
            predictions.append(int(prediction))
            ground_truths.append(int(ground_truth))
    
    f1 = f1_score(ground_truths, predictions, zero_division=0)
    
    return f1


def calculate_binary_metrics(csv_file):
    precision=calculate_precision(csv_file)
    recall=calculate_recall(csv_file)
    f1=calculate_f1(csv_file)

    print("Precision score, ", precision)
    print("Recall score, ", recall)
    print("F1 score score, ", f1)


calculate_binary_metrics('/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/PhysNet_NeuralTexturesNeuralTextures_ADAM_LR=0.001_LRReducer_16_frames_0.5binary_0.5rPPG_scaling_multi_model_continue_from_epoch_16_epoch7/predictions.csv')