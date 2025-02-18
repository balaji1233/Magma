import torch.distributed as dist

# get rank and world size
rank = dist.get_rank()
world_size = dist.get_world_size()