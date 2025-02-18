import torch
from mpi4py import MPI

comm = MPI.COMM_WORLD

# get the number of devices
def get_world_size():
    return comm.Get_size()

# get the device gpu rank for multiple gpus
def get_device_rank():
    return comm.Get_rank()

# get the default device
def get_default_device():
    return torch.device(f"cuda:{get_device_rank()}" if torch.cuda.is_available() else "cpu")

def rank_print(*args, **kwargs):
    if get_device_rank() == 0:
        print(*args, **kwargs)