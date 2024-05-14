import csv

# text file containing the data
data_file_path = 'rppgpredictions_16frames_trainset2.txt' 

# path to CSV file
csv_file_path = 'rppgpredictions_16frames_train_final.csv'

csv_headers = ["Batch Index", "frame_path", "rPPG_values"]

with open(data_file_path, 'r') as txt_file, open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)
    
    for line in txt_file:
        batch_index, frame_path, rppg_values_str = line.strip().split(',', 2)
        frame_path = frame_path.strip()
        rppg_values_str = rppg_values_str.strip().replace(' ', ', ')
        writer.writerow([batch_index, frame_path, rppg_values_str])

print(f"Data added to {csv_file_path} successfully.")
