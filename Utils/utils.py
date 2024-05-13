import cv2
import os
import csv

def calculate_accuracy(pred_labels, labels):
    acc=0.0
    correct = 0.0
    for p, g in zip(pred_labels, labels):
        if p == g:
            correct += 1
    acc = 100 * correct/len(pred_labels)
    
    return acc

def save_predictions(pred_labels, gt_labels, test_filepaths, modelname):    
    dir_name=os.path.join('/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/', modelname)
    os.makedirs(dir_name, exist_ok=True)
    
    csv_path = os.path.join(dir_name, 'predictions.csv')

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filepath', 'Prediction', 'Ground Truth'])
        for p, g, f in zip(pred_labels, gt_labels, test_filepaths):
            writer.writerow([f, p, g])
    print(f'Predictions saved to {csv_path}')

def get_missclassified_samples(pred_labels, gt_labels, test_filepaths):
    misclassified_info=[]
    for p,g,f in zip(pred_labels, gt_labels, test_filepaths):
        if p != g:
            misclassified_info.append(f)
    return misclassified_info

def save_misclassified_samples(filepaths_misclassifed_samples, config):
    for filepath_chunk in filepaths_misclassifed_samples:
        parts = filepath_chunk.split('/')
        second_last_number = parts[-2]
        img = cv2.imread(filepath_chunk)
        save_dir = f"/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/{config.TRAIN.MODEL_FILE_NAME}/{second_last_number}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, parts[-1])
        success = cv2.imwrite(save_path, img)
        print(f"Save successful: {success}")
