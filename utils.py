import os
import sys
import argparse
import logging
from scipy.io.wavfile import read
import torch
from torch.nn import functional as F
import tqdm
import yaml

MATPLOTLIB_FLAG = False
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging

os.makedirs('chkpt/sovits5.0', exist_ok=True)

with open('configs/base.yaml', 'r') as f:
  config = yaml.safe_load(f)

keep_ckpts = config['log']['keep_ckpts']


def clean_checkpoints(path_to_models='chkpt/sovits5.0/', n_ckpts_to_keep=keep_ckpts, sort_by_time=True):
  """Freeing up space by deleting saved ckpts

  Arguments:
  path_to_models    --  Path to the model directory
  n_ckpts_to_keep   --  Number of ckpts to keep, excluding sovits5.0_0.pth
                        If n_ckpts_to_keep == 0, do not delete any ckpts
  sort_by_time      --  True -> chronologically delete ckpts
                        False -> lexicographically delete ckpts
  """
  assert isinstance(n_ckpts_to_keep, int) and n_ckpts_to_keep >= 0
  ckpts_files = [f for f in os.listdir(path_to_models) if os.path.isfile(os.path.join(path_to_models, f))]
  name_key = (lambda _f: int(re.compile('sovits5.0_(\d+)\.pt').match(_f).group(1)))
  time_key = (lambda _f: os.path.getmtime(os.path.join(path_to_models, _f)))
  sort_key = time_key if sort_by_time else name_key
  x_sorted = lambda _x: sorted([f for f in ckpts_files if f.startswith(_x) and not f.endswith('sovits5.0_0.pth')], key=sort_key)
  if n_ckpts_to_keep == 0:
      to_del = []
  else:
      to_del = [os.path.join(path_to_models, fn) for fn in x_sorted('sovits5.0')[:-n_ckpts_to_keep]]
  del_info = lambda fn: logger.info(f".. Free up space by deleting ckpt {fn}")
  del_routine = lambda x: [os.remove(x), del_info(x)]
  rs = [del_routine(fn) for fn in to_del]
clean_checkpoints()


