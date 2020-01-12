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

    self.args=self.parse_args()
