import torch
import torch.nn as nn
import torch.optim as optim

class MultiPhysNetModel(nn.Module):
    def __init__(self, physnet_model1, physnet_model2, mlp1_output_dim, mlp2_output_dim):
        super(MultiPhysNetModel, self).__init__()
        
        self.physnet1 = physnet_model1
        self.physnet2 = physnet_model2
        
        self.concatenated_output_size = self.physnet1.frames + self.physnet1.frames
        
        # MLP heads
        self.mlp_head1 = nn.Sequential(
            nn.Linear(self.concatenated_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, mlp1_output_dim)
        )
        
        self.mlp_head2 = nn.Sequential(
            nn.Linear(self.concatenated_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, mlp2_output_dim)
        )

    def forward(self, video_chunk1, video_chunk2):
        video_chunk1 = video_chunk1.permute(0, 2, 1, 3, 4)
        video_chunk2 = video_chunk2.permute(0, 2, 1, 3, 4)

        _, embedding1, _, _, _ = self.physnet1(video_chunk1)
        _, embedding2, _, _, _ = self.physnet2(video_chunk2)
        
        # concatenating the outputs
        concatenated_outputs = torch.cat((embedding1, embedding2), dim=1)
        # [6,2,16,1,1]
        concatenated_outputs = concatenated_outputs.squeeze()
        # [6,2,16]
        # [6,32]
        concatenated_outputs=concatenated_outputs.reshape(-1, self.concatenated_output_size)

        # passing through the MLP heads
        mlp1_output = self.mlp_head1(concatenated_outputs)
        mlp2_output = self.mlp_head2(concatenated_outputs)
        
        return mlp1_output, mlp2_output