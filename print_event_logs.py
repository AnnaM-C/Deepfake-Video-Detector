import tensorflow as tf
import os

logdir = "/vol/research/DeepFakeDet/rPPG-Toolbox/runs/exp/logs/output_2/PhysNet/PhysNet_NeuralTextures_SGD_LR=0.001_LRReducer_32_frames_new_preprocessed_ds_real_paths_more_frames_6"

for file in os.listdir(logdir):
    if file.startswith("events.out.tfevents"):
        filepath = os.path.join(logdir, file)
        for e in tf.compat.v1.train.summary_iterator(filepath):
            for v in e.summary.value:
                print("Step:", e.step, "Tag:", v.tag, "Value:", v.simple_value)
                
