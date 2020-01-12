import csv
import numpy as np
import pandas as pd
from random import shuffle

# ================================================================================
import torch.utils.data as data

# ================================================================================
unique_characters_for_protein={
  "A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,
  "Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}

length_of_unique_characters_for_protein=len(unique_characters_for_protein)
# print("length_of_unique_characters_for_protein",length_of_unique_characters_for_protein)
# length_of_unique_characters_for_protein 25

# ================================================================================
unique_characters_for_can_smiles={
  "#":1,"%":2,")":3,"(":4,"+":5,"-":6,".":7,"1":8,"0":9,"3":10,"2":11,"5":12,"4":13,"7":14,"6":15,"9":16,"8":17,"=":18,
  "A":19,"C":20,"B":21,"E":22,"D":23,"G":24,"F":25,"I":26,"H":27,"K":28,"M":29,"L":30,"O":31,"N":32,"P":33,"S":34,
  "R":35,"U":36,"T":37,"W":38,"V":39,"Y":40,"[":41,"Z":42,"]":43,"_":44,"a":45,"c":46,"b":47,"e":48,"d":49,"g":50,
  "f":51,"i":52,"h":53,"m":54,"l":55,"o":56,"n":57,"s":58,"r":59,"u":60,"t":61,"y":62}

length_of_unique_characters_for_can_smiles=len(unique_characters_for_can_smiles)
# print("length_of_unique_characters_for_can_smiles",length_of_unique_characters_for_can_smiles)
# length_of_unique_characters_for_can_smiles 62

# ================================================================================
unique_characters_for_iso_smiles={
  "#":29,"%":30,")":31,"(":1,"+":32,"-":33,"/":34,".":2,"1":35,"0":3,"3":36,"2":4,"5":37,"4":5,"7":38,"6":6,"9":39,
  "8":7,"=":40,"A":41,"@":8,"C":42,"B":9,"E":43,"D":10,"G":44,"F":11,"I":45,"H":12,"K":46,"M":47,"L":13,"O":48,"N":14,
  "P":15,"S":49,"R":16,"U":50,"T":17,"W":51,"V":18,"Y":52,"[":53,"Z":19,"]":54,"\\":20,"a":55,"c":56,"b":21,"e":57,
  "d":22,"g":58,"f":23,"i":59,"h":24,"m":60,"l":25,"o":61,"n":26,"s":62,"r":27,"u":63,"t":28,"y":64}

length_of_unique_characters_for_iso_smiles=len(unique_characters_for_iso_smiles)
# print("length_of_unique_characters_for_iso_smiles",length_of_unique_characters_for_iso_smiles)
# length_of_unique_characters_for_iso_smiles 64

# ================================================================================
class DeepDTA_Dataset(data.Dataset):
  def __init__(self,args):
    print("args",args)

    # print("single_train_k",single_train_k)
    # [['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png\n'
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_green.png\n'
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_red.png\n'
    #   '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_yellow.png\n']

    # print("single_train_lbl_k",single_train_lbl_k)
    # [['000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0' '7 1 2 0']
    #  ['000a9596-bbc4-11e8-b2bc-ac1f6b6435d0' '5']
    #  ['000c99ba-bba4-11e8-b2b9-ac1f6b6435d0' '1']
    #  ...
    #  ['ffe55eba-bbba-11e8-b2ba-ac1f6b6435d0' '5 0']
    #  ['ffe61798-bbc3-11e8-b2bc-ac1f6b6435d0' '19 23']
    #  ['ffe8cf0c-bba9-11e8-b2ba-ac1f6b6435d0' '18']]

    # print("single_train_lbl_k",single_train_lbl_k[:,1])
    # ['7 1 2 0' '5' '1' ... '5 0' '19 23' '18']

    # set_of_labels=[list(map(int,one_label_set.split(" "))) for one_label_set in single_train_lbl_k[:,1]]
    # print("set_of_labels",set_of_labels)
    # [[7, 1, 2, 0],

    # zipped=np.array(list(zip(single_train_k,single_train_lbl_k[:,1])))
    # print("zipped",zipped)
    # [[array(['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png',
    #          '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_green.png',
    #          '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_red.png',
    #          '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_yellow.png'],dtype='<U140')
    #  '7 1 2 0']

    # ================================================================================
    # zipped_new=[[list(one_protein_pair[0]),one_protein_pair[1]] for one_protein_pair in zipped]
    # print("zipped_new",zipped_new)
    # [[['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_green.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_red.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_yellow.png'],
    #  '7 1 2 0'],
 
    # took_time_min 0:00:00.130517

    # shuffle(zipped_new)
    # print("zipped_new",zipped_new)
    # [[['/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/e6e2c6c0-bbae-11e8-b2ba-ac1f6b6435d0_blue.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/e6e2c6c0-bbae-11e8-b2ba-ac1f6b6435d0_green.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/e6e2c6c0-bbae-11e8-b2ba-ac1f6b6435d0_red.png',
    #    '/mnt/1T-5e7/mycodehtml/bio_health/Kaggle/human-protein-atlas-image-classification/Data/train/e6e2c6c0-bbae-11e8-b2ba-ac1f6b6435d0_yellow.png'],
    #  '3 0'],

    # self.trn_pairs=zipped_new

    # ================================================================================
    # instance of argument 
    # self.args=args

    # ================================================================================
    # self.nb_trn_imgs=len(single_train_k)
    # 20714

  # ================================================================================
  def __len__(self):
    # return self.nb_trn_imgs
    pass

  # ================================================================================
  def __getitem__(self,idx):
    # one_pair=self.trn_pairs[idx]
    # return one_pair
    pass
