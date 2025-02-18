import glob
import os

traces_dir = "/home/jianwyan/projects/ProjectWillow/azureblobs/vlpdatasets/open-x-traces"

# find all .pth files in the traces directory recursively
pth_files = glob.glob(os.path.join(traces_dir, "**/*.pth"), recursive=True)

import pdb; pdb.set_trace()
