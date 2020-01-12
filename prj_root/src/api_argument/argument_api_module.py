import datetime
import sys,os,copy,argparse

class Argument_API_class(argparse.ArgumentParser):
  def __init__(self):
    super(Argument_API_class,self).__init__()
    self.add_argument("--start_mode",help="preanalyze_data,train")
    self.add_argument("--epoch")
    self.add_argument("--batch_size")
    self.add_argument("--task_mode",default=True)
    self.add_argument("--train_method",help="train_by_custom_net,train_by_transfer_learning_using_resnet")
    self.add_argument("--max_length_of_smiles_sequence",default=100,help="integer")
    self.add_argument("--max_length_of_protein_sequence",default=1000,help="integer")
    self.add_argument("--embedding_vector_dimension_for_smiles_and_protein",default=128,help="integer")
    self.add_argument("--dataset_path",default="../Data/kiba",help="data directory")
    self.add_argument("--with_label",default=True)
    self.add_argument("--problem_type",default=1)
    self.add_argument("--num_windows",nargs="*",type=int,default=[35,40,50,60,70,80,90])
    self.add_argument("--smi_window_lengths",nargs="*",type=int,default=[35,40,50,60,70,80,90])
    self.add_argument("--seq_window_lengths",nargs="*",type=int,default=[35,40,50,60,70,80,90])
    # self.add_argument("--smi_window_lengths")
    # self.add_argument("--seq_window_lengths")

    self.args=self.parse_args()
