import torch
import pdb
from time import sleep

# pdb.set_trace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.rand((1024,1024*187)).to(device)
xs = []

while 1:
    try:
        xs.append(torch.rand((1024,1024)).to(device))
        print(len(xs))
    except:
        sleep(5)
        pass

