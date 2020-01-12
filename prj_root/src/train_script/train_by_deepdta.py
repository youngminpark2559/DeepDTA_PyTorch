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

from src.networks import network_deepdta as network_deepdta
from src.utils_for_dataset import dataset_deepdta as dataset_deepdta

# ================================================================================
def train(args):

  deepdta_obj=network_deepdta.DeepDTA(args).cuda()
  # print("deepdta_obj",deepdta_obj)

  for one_ep in range(int(args.epoch)): # @ Iterates all epochs

    # ================================================================================
    # c dataset_inst_trn: dataset instance of tumor
    deepdta_dataset_obj=dataset_deepdta.DeepDTA_Dataset(args)
    # print("deepdta_dataset_obj",deepdta_dataset_obj)

    # ================================================================================
    # c dataloader_trn: create dataloader
    deepdta_dataloader_obj=torch.utils.data.DataLoader(dataset=deepdta_dataset_obj,batch_size=int(args.batch_size),shuffle=False,num_workers=3)

    # ================================================================================
    for idx,data in enumerate(deepdta_dataloader_obj):
      # print("idx",idx)
      # print("len(data)",len(data))
      # idx 0
      # len(data) 3
      
      batch_drugs=data[0].long().cuda()
      batch_proteins=data[1].long().cuda()
      batch_affinities=data[2].long().cuda()
      # print("batch_drugs",batch_drugs)
      # print("batch_proteins",batch_proteins)
      # print("batch_affinities",batch_affinities)
      # print("batch_drugs",batch_drugs.shape)
      # print("batch_proteins",batch_proteins.shape)
      # print("batch_affinities",batch_affinities.shape)
      # batch_drugs torch.Size([256, 100])
      # batch_proteins torch.Size([256, 1000])
      # batch_affinities torch.Size([256])

      deepdta_obj(batch_drugs,batch_proteins)
      
