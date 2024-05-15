import matplotlib.pyplot as plt
import numpy as np
import os

total_frames = 270
chunk_size = 32
frame_step_non_overlapping = chunk_size
frame_step_overlapping = 2 

non_overlapping_batches = total_frames // chunk_size
overlapping_batches = (total_frames - chunk_size) // frame_step_overlapping + 1

batch_labels = ['Non-overlapping\n{} batches'.format(non_overlapping_batches),
                'Overlapping\n{} batches'.format(overlapping_batches)]

bar_heights_exact = [non_overlapping_batches, overlapping_batches]

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(batch_labels, bar_heights_exact, color=['blue', 'orange'])

ax.set_title('Comparison of Batches with Non-overlapping vs. Overlapping Frames')
ax.set_xlabel('Number of Batches')
ax.set_ylabel('Frame Overlap Type')

ax.set_xticks(np.arange(0, max(bar_heights_exact)+1, 16))

plt.tight_layout()

save_directory = '/vol/research/DeepFakeDet/rPPG-Toolbox/results_0/graphs/samples'

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

plt.savefig(f'{save_directory}/batch_comparison.png')