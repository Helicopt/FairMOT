from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset
from .dataset.rel import RelJointDataset


def get_dataset(dataset, task):
  if task == 'mot':
    if dataset == 'vrel':
      return RelJointDataset
    return JointDataset
  else:
    return None
  
