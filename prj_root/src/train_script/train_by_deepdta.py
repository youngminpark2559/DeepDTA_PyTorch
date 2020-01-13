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
criterion=nn.MSELoss()

# ================================================================================
def train(args):

  deepdta_obj=network_deepdta.DeepDTA(args).cuda()
  # print("deepdta_obj",deepdta_obj)

  optimizer=torch.optim.Adam(deepdta_obj.parameters(),lr=0.01)

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
      batch_affinities=data[2].float().cuda()
      # print("batch_drugs",batch_drugs)
      # print("batch_proteins",batch_proteins)
      # print("batch_affinities",batch_affinities)
      # print("batch_drugs",batch_drugs.shape)
      # print("batch_proteins",batch_proteins.shape)
      # print("batch_affinities",batch_affinities.shape)
      # batch_drugs torch.Size([256, 100])
      # batch_proteins torch.Size([256, 1000])
      # batch_affinities torch.Size([256])

      if batch_affinities.shape[0]!=256:
        continue

      predicted_value=deepdta_obj(batch_drugs,batch_proteins)
      # print("predicted_value",predicted_value.shape)
      # predicted_value tensor([0.0069], device='cuda:0', grad_fn=<AddBackward0>)

      loss_value=criterion(predicted_value,batch_affinities)
      print("loss_value",loss_value)
      # loss_value tensor(137.5635, device='cuda:0', grad_fn=<MseLossBackward>)

      optimizer.zero_grad()

      loss_value.backward()
  
      optimizer.step()

    print("one epoch end")
