from src.networks import network_deepdta as network_deepdta
from src.utils_for_dataset import dataset_deepdta as dataset_deepdta

# ================================================================================
def train(args):

  deepdta_dataset_obj=dataset_deepdta.DeepDTA_Dataset(args)
  print("deepdta_dataset_obj",deepdta_dataset_obj)
  



  deepdta_obj=network_deepdta.DeepDTA(args)
  print("deepdta_obj",deepdta_obj)
  afaf
