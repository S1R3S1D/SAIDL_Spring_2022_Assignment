import torch
import numpy as np
import torch.nn as nn

#Modified normalized temperature scaled cross entropy loss function
class ntxent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(ntxent, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("neg_eye", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.register_buffer("temperature", torch.tensor(temperature))
    def forward(self, emb_i, emb_j):

        z_i = F.normalize(emb_i)
        z_j = F.normalize(emb_j)

        z = torch.cat([z_i, z_j], 0)

        sim_mat = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(sim_mat, self.batch_size)
        sim_ji = torch.diag(sim_mat, -self.batch_size)

        positives = torch.cat([sim_ij, sim_ji], 0)

        numerator = torch.exp(positives/self.temperature)
        denominator = self.neg_eye*torch.exp(sim_mat/self.temperature)

        loss_partial = -torch.log(numerator/(torch.sum(denominator, 0)-numerator)) #The modification : subtracting components of similarity from the denominator

        loss = torch.sum(loss_partial)/(2*self.batch_size)

        return loss
