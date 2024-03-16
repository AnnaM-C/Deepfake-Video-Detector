#import tensorflow as tf
#from tensorflow.core.util import event_pb2
#from tensorflow.python.lib.io import tf_record

# Path to your event file, typically something like './logs/train/events.out.tfevents.xxxxxx'
#event_file_path = '/vol/research/DeepFakeDet/rPPG-Toolbox/events.out.tfevents.1710248578.tr00564-156405.0-aisurrey24.surrey.ac.uk.17.1'

# Create a TFRecordDataset
#dataset = tf.data.TFRecordDataset(event_file_path)

# Function to decode each record
#def decode(record):
#    event = event_pb2.Event()  # Create an empty Event
#    event.ParseFromString(record.numpy())  # Parse the record into the event
#    return event

# print(dataset)

# Iterate through records and process them
#for raw_record in dataset:
    # print(raw_record)
#    event = decode(raw_record)
    # print(event)
#    for value in event.summary.value:
        # Here you can handle each summary value (this is just an example)
#        print("Tag:", value.tag)
#        if value.HasField('simple_value'):
#            print(f"{event.step}:", "Simple value:", value.simple_value)

import tensorflow as tf
import os

logdir = "/vol/research/DeepFakeDet/rPPG-Toolbox/runs/exp/logs/output_2/PhysNet/PhysNet_NeuralTextures_SGD_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_6"

for file in os.listdir(logdir):
    if file.startswith("events.out.tfevents"):
        filepath = os.path.join(logdir, file)
        for e in tf.compat.v1.train.summary_iterator(filepath):
            for v in e.summary.value:
                print("Step:", e.step, "Tag:", v.tag, "Value:", v.simple_value)
                
