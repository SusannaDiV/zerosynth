import torch
import torch.nn as nn
import torch.nn.functional as F
class ShapePretrainingModel(nn.Module):
    def __init__(self, encoder, d_model, no_shape=False, no_trans=False, no_rotat=False):
        super().__init__()
        self.encoder = encoder
        self.no_shape = no_shape
        self.no_trans = no_trans
        self.no_rotat = no_rotat

    def forward(self, shape_patches): #, input_frag_idx, input_frag_trans, input_frag_r_mat):
        memory = self.encoder(shape_patches)
        '''
        if self.no_shape:
            memory = torch.zeros_like(memory)
        if self.no_trans:
            input_frag_trans = torch.zeros_like(input_frag_trans)
        if self.no_rotat:
            input_frag_r_mat = torch.zeros_like(input_frag_r_mat)
        # Decoder logic would go here
        '''
        return memory