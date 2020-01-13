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
    
    # ================================================================================
    # @ Embedding

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

    # ================================================================================
    # @ Convolution for smiles
    
    self.conv1_for_smiles=nn.Sequential(
      nn.Conv1d(100,128,kernel_size=4))
    self.conv2_for_smiles=nn.Sequential(
      nn.Conv1d(128,128,kernel_size=4))
    self.conv3_for_smiles=nn.Sequential(
      nn.Conv1d(128,128,kernel_size=4))

    # ================================================================================
    self.conv1_for_proteins=nn.Sequential(
      nn.Conv1d(1000,128,kernel_size=4))
    self.conv2_for_proteins=nn.Sequential(
      nn.Conv1d(128,128,kernel_size=4))
    self.conv3_for_proteins=nn.Sequential(
      nn.Conv1d(128,128,kernel_size=4))

    # ================================================================================
    # @ Fully connected layer

    self.fc1=nn.Linear(60928,1024)
    self.relu1=nn.ReLU()
    self.dropout1=torch.nn.Dropout(p=0.1)
    self.fc2=nn.Linear(1024,512)
    self.relu2=nn.ReLU()
    self.dropout2=torch.nn.Dropout(p=0.1)
    self.fc3=nn.Linear(512,512)
    self.relu3=nn.ReLU()

    # ================================================================================
    # @ Output layer

    self.fc_out=nn.Linear(512,1)

  def forward(self,batch_drugs,batch_proteins):

    # ================================================================================
    # @ Embed smiles and protein data

    embedded_smiles_data=self.embedding_for_smiles(batch_drugs)
    embedded_protein_data=self.embedding_for_protein(batch_proteins)
    # print("embedded_smiles_data",embedded_smiles_data)
    # print("embedded_protein_data",embedded_protein_data)
    # print("embedded_smiles_data",embedded_smiles_data.shape)
    # print("embedded_protein_data",embedded_protein_data.shape)
    # embedded_smiles_data torch.Size([256, 100, 128])
    # embedded_protein_data torch.Size([256, 1000, 128])
    
    # ================================================================================
    # @ Perform convolution for "smiles data" and "protein data"

    smiles_after_conv=self.conv1_for_smiles(embedded_smiles_data)
    smiles_after_conv=self.conv2_for_smiles(smiles_after_conv)
    smiles_after_conv=self.conv3_for_smiles(smiles_after_conv)
    # print("smiles_after_conv",smiles_after_conv.shape)
    # torch.Size([256, 128, 119])
    smiles_after_max_pool, _  = torch.max(smiles_after_conv,1)
    # smiles_after_max_pool=F.max_pool2d(smiles_after_conv,kernel_size=smiles_after_conv.size()[2:])
    # print("smiles_after_max_pool",smiles_after_max_pool)
    # print("smiles_after_max_pool",smiles_after_max_pool.shape)
    # smiles_after_max_pool torch.Size([256, 119])

    # ================================================================================
    protein_after_conv=self.conv1_for_proteins(embedded_protein_data)
    protein_after_conv=self.conv2_for_proteins(protein_after_conv)
    protein_after_conv=self.conv3_for_proteins(protein_after_conv)
    protein_after_max_pool, _  = torch.max(protein_after_conv,1)
    # print("protein_after_max_pool",protein_after_max_pool)
    # print("protein_after_max_pool",protein_after_max_pool.shape)
    # protein_after_max_pool torch.Size([256, 119])

    # ================================================================================
    concatenated_tensor=torch.cat((smiles_after_max_pool,protein_after_max_pool),-1)
    # print("concatenated_tensor",concatenated_tensor.shape)
    # concatenated_tensor torch.Size([256, 238])

    # ================================================================================
    flattened_tensor=concatenated_tensor.view(-1)
    # print("flattened_tensor",flattened_tensor.shape)
    # flattened_tensor torch.Size([60928])

    # ================================================================================
    out_from_fc1=self.fc1(flattened_tensor)
    out_from_relu1=self.relu1(out_from_fc1)
    out_from_dropout1=self.dropout1(out_from_relu1)
    out_from_fc2=self.fc2(out_from_dropout1)
    out_from_relu2=self.relu2(out_from_fc2)
    out_from_dropout2=self.dropout2(out_from_relu2)
    out_from_fc3=self.fc3(out_from_dropout2)
    out_from_relu3=self.relu3(out_from_fc3)
    out_from_fc_out=self.fc_out(out_from_relu3)
    # print("out_from_fc_out",out_from_fc_out.shape)
    # out_from_fc_out torch.Size([1])
    # print("out_from_fc_out",out_from_fc_out)
    # out_from_fc_out tensor([0.0098], device='cuda:0', grad_fn=<AddBackward0>)

    return x
