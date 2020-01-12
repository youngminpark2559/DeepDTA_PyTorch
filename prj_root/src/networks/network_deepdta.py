import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets,models,transforms

# ================================================================================
def init_weights(m):
  if type(m) == nn.Linear or\
     type(m) == nn.Conv2d or\
     type(m) == nn.ConvTranspose2d:
    torch.nn.init.xavier_uniform_(m.weight.data)
    # torch.nn.init.xavier_uniform_(m.bias)
    # m.bias.data.fill_(0.01)

# ================================================================================
class DeepDTA(nn.Module):

  def __init__(self,args):
    super(DeepDTA,self).__init__()
    
    self.embedding_for_smiles = nn.Embedding(int(args.max_length_of_smiles_sequence),int(args.embedding_vector_dimension_for_smiles_and_protein))
    self.embedding_for_protein = nn.Embedding(int(args.max_length_of_protein_sequence),int(args.embedding_vector_dimension_for_smiles_and_protein))
    # print("self.embedding_for_smiles.weight",self.embedding_for_smiles.weight)
    # print("self.embedding_for_protein.weight",self.embedding_for_protein.weight)
    # self.embedding_for_smiles.weight Parameter containing:
    # tensor([[-0.5787, -0.3754,  0.1816,  ...,  0.2408,  0.9821, -0.3493],
    #         [-1.0835, -1.5485,  0.3644,  ..., -1.1137,  0.3518,  0.4952],
    #         [ 0.2292, -0.1921,  0.1039,  ..., -1.0953, -0.0255, -0.2318],
    #         ...,
    #         [-1.0763,  1.0572, -0.1679,  ...,  0.4428,  1.2395, -0.4269],
    #         [-0.1436,  0.0261, -0.5508,  ..., -0.5458, -0.1447, -1.2411],
    #         [-0.8655, -1.8142,  2.0100,  ..., -1.3593,  0.9579, -1.2367]],
    #       requires_grad=True)
    # self.embedding_for_protein.weight Parameter containing:
    # tensor([[ 0.2875,  0.3775, -0.0666,  ..., -0.5810, -0.0225, -0.7573],
    #         [-1.0012, -1.1658, -0.2750,  ..., -0.3258,  0.4019, -0.0047],
    #         [ 0.0569, -0.3370,  0.5988,  ..., -1.1315,  1.5031,  0.7707],
    #         ...,
    #         [-0.5997, -0.7764, -0.0220,  ..., -0.6401, -0.6737,  0.4884],
    #         [-0.0604,  1.3284, -2.0070,  ...,  0.9760, -1.5040,  0.8870],
    #         [ 0.7017, -0.2623, -1.1540,  ..., -1.7957, -0.6562,  0.7233]],
    #       requires_grad=True)

    # print("self.embedding_for_smiles",self.embedding_for_smiles.weight.shape)
    # print("self.embedding_for_protein",self.embedding_for_protein.weight.shape)
    # self.embedding_for_smiles torch.Size([100, 128])
    # self.embedding_for_protein torch.Size([1000, 128])

  def forward(self,smiles_data,protein_data):
    
    # ================================================================================
    # @ Embed smiles and protein data

    embedded_smiles_data=self.embedding_for_smiles(smiles_data)
    embedded_protein_data=self.embedding_for_protein(protein_data)

    # ================================================================================

    


    return x
