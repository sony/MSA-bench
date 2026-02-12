#! python

import torch
import torch.nn as nn
import torch.nn.functional as F

##
## Loss Function
##
# Binary Cross Entropy
class BinaryCrossEntropy_with_logits(nn.Module):
    def __init__(self, pos_weight=1.0, reduction='mean'):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)
        self.reduction = reduction

    # predict should NOT be processed w/ sigmoid()
    def forward(self, predict, target):
        return F.binary_cross_entropy_with_logits(predict, target, pos_weight=self.pos_weight, reduction=self.reduction)


##
## training
##
def train(model, iterator, optimizer,
          criterion_structure_name, criterion_structure_boundary,
          device):

    model.train()
    epoch_loss = 0

    for i, batch_data in enumerate(iterator):
        # input
        input_embed = batch_data['input']['embed'].float().to(device, non_blocking=True)
        # input_embed: [B, Te, Ze]
        #print('(1) input_embed: '+str(input_embed.shape))

        # label
        label_structure_name = batch_data['label']['structure_name'].long().to(device, non_blocking=True)
        label_structure_boundary = batch_data['label']['structure_boundary'].float().to(device, non_blocking=True)
        # label_structure_name: [B, Ts, n_structure_name]
        # label_structure_root: [B, Ts, n_structure_boundary]
        #print('(2) label_structure_name: '+str(label_structure_name.shape))
        #print('(2) label_structure_boundary: '+str(label_structure_boundary.shape))

        # mask
        batch_data['mask'] = batch_data['mask'].to(device, non_blocking=True)
        #print('(3) mask: '+str(batch_data['mask'].shape))

        optimizer.zero_grad()
        output_structure_name, output_structure_boundary = model(input_embed)
        # output_structure_name: [B, Ts, n_structure_name]
        # output_structure_root: [B, Ts, n_structure_boundary]
        #print('(4) output_structure_name: '+str(output_structure_name.shape))
        #print('(4) output_structure_boundary: '+str(output_structure_boundary.shape))

        # structure
        label_structure_name = label_structure_name.view(-1)
        output_structure_name_dim = output_structure_name.shape[-1]
        output_structure_name = output_structure_name.view(-1, output_structure_name_dim)
        #print('(5) label_structure_name: '+str(label_structure_name.shape))
        #print('(5) output_structure_name: '+str(output_structure_name.shape))

        loss_structure_name = criterion_structure_name(output_structure_name, label_structure_name)
        loss_structure_boundary = criterion_structure_boundary(output_structure_boundary, label_structure_boundary)
        #print('(6) loss_structure_name: '+str(loss_structure_name.shape))
        #print('(6) loss_structure_boundary: '+str(loss_structure_boundary.shape))

        loss_structure_name *= batch_data['mask'].view(-1)
        loss_structure_boundary *= batch_data['mask']
        loss_structure_name = torch.nanmean(loss_structure_name)
        loss_structure_boundary = torch.nanmean(loss_structure_boundary)

        loss = loss_structure_name + loss_structure_boundary
        #print('(7) loss               : '+str(loss.item()))
        #print('(7) _structure_name    : '+str(loss_structure_name.item()))
        #print('(7) _structure_boundary: '+str(loss_structure_boundary.item()))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss, len(iterator)


##
## validation
##
def valid(model, iterator,
          criterion_structure_name, criterion_structure_boundary,
          device):

    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch_data in enumerate(iterator):
            # input
            input_embed = batch_data['input']['embed'].float().to(device, non_blocking=True)
            # label
            label_structure_name = batch_data['label']['structure_name'].long().to(device, non_blocking=True)
            label_structure_boundary = batch_data['label']['structure_boundary'].float().to(device, non_blocking=True)
            # mask
            batch_data['mask'] = batch_data['mask'].to(device, non_blocking=True)

            output_structure_name, output_structure_boundary = model(input_embed)

            label_structure_name = label_structure_name.view(-1)
            output_structure_name_dim = output_structure_name.shape[-1]
            output_structure_name = output_structure_name.view(-1, output_structure_name_dim)
            loss_structure_name = criterion_structure_name(output_structure_name, label_structure_name)
            loss_structure_boundary = criterion_structure_boundary(output_structure_boundary, label_structure_boundary)

            loss_structure_name *= batch_data['mask'].view(-1)
            loss_structure_boundary *= batch_data['mask']
            loss_structure_name = torch.nanmean(loss_structure_name)
            loss_structure_boundary = torch.nanmean(loss_structure_boundary)

            loss = loss_structure_name + loss_structure_boundary

            epoch_loss += loss.item()

    return epoch_loss, len(iterator)
