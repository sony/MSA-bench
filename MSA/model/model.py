#! python

import torch
import torch.nn as nn

class MODEL_A(nn.Module):
    def __init__(self, parameter):
        super().__init__()

        # head for structure
        self.head_structure = HEAD_STRUCTURE(
            parameter['Z'],
            parameter['structure']['name'],
            parameter['structure']['boundary']
        )

    def forward(self, input_embed):
        # input_embed: [B, Ti, Z] (8, 60, 768)
        #print('MODEL_A(0) input_embed: '+str(input_embed.shape))

        # structure head
        output_structure_name, output_structure_boundary = self.head_structure(input_embed)
        # output_structure_name: [B, To, n_structure_name] (8, 60, 7)
        # output_structure_boundary: [B, To] (8, 60)
        #print('MODEL_A(1) output_structure_name: '+str(output_structure_name.shape))
        #print('MODEL_A(1) output_structure_boundary: '+str(output_structure_boundary.shape))

        return output_structure_name, output_structure_boundary


class MODEL_B(nn.Module):
    def __init__(self, parameter):
        super().__init__()

        # downsampling
        self.downsampling = DOWNSAMPLE(
            parameter['To']
        )

        # head for structure
        self.head_structure = HEAD_STRUCTURE(
            parameter['Z'],
            parameter['structure']['name'],
            parameter['structure']['boundary']
        )

    def forward(self, input_embed):
        # input_embed: [B, Ti, Z] (8, 800, 1024)
        #print('MODEL_B(0) input_embed: '+str(input_embed.shape))

        # downsampling
        z = self.downsampling(input_embed)
        # z: [B, To, Z] (8, 60, 1024)
        #print('MODEL_B(1): z: '+str(z.shape))

        # structure head
        output_structure_name, output_structure_boundary = self.head_structure(z)
        # output_structure_name: [B, To, n_structure_name] (8, 60, 7)
        # output_structure_boundary: [B, To] (8, 60)
        #print('MODEL_B(2) output_structure_name: '+str(output_structure_name.shape))
        #print('MODEL_B(2) output_structure_boundary: '+str(output_structure_boundary.shape))

        return output_structure_name, output_structure_boundary


# downsampling
class DOWNSAMPLE(nn.Module):
    def __init__(self, To):
        super().__init__()

        self.To = To
        self.downsample = nn.AdaptiveAvgPool1d(To)

    def forward(self, input_embed):
        # input_dim: [B, Ti, Z]
        #print('DOWNSAMPLE(0): input_embed: '+str(input_embed.shape))

        ret = input_embed.shape
        z = input_embed.permute(0, 2, 1).contiguous().reshape(ret[0]*ret[2], ret[1])
        # z: [B*Z, Ti]
        #print('DOWNSAMPLE(1): z: '+str(z.shape))

        z = self.downsample(z)
        # z: [B*Z, To]
        #print('DOWNSAMPLE(2): z: '+str(z.shape))

        z = z.reshape(ret[0], ret[2], self.To).permute(0, 2, 1).contiguous()
        # z: [B, To, Z]
        #print('DOWNSAMPLE(3): z: '+str(z.shape))

        return z


# structure head
class HEAD_STRUCTURE(nn.Module):
    def __init__(self, Z, n_structure_name, n_structure_boundary):
        super().__init__()

        self.n_structure_name = n_structure_name
        self.n_structure_boundary = n_structure_boundary
        self.linear = nn.Linear(Z, n_structure_name+n_structure_boundary)

    def forward(self, z):
        # z: [B, To, Z]
        #print('HEAD_STRUCTURE(0) z: '+str(z.shape))

        output = self.linear(z)
        # output: [B, To, n_structure_name+n_structure_boundary]
        #print('HEAD_STRUCTURE(1) output: '+str(output.shape))

        output_structure_name = output[:,:,:self.n_structure_name]
        # output_structure_name: [B, To, n_structure_name]
        output_structure_boundary = output[:,:,-self.n_structure_boundary:].squeeze(2)
        # output_structure_boundary: [B, To]
        #print('HEAD_STRUCTURE(2) output_structure_name: '+str(output_structure_name.shape))
        #print('HEAD_STRUCTURE(2) output_structure_boundary: '+str(output_structure_boundary.shape))

        return output_structure_name, output_structure_boundary
