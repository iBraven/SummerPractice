#Main
import torch as th
print(f'version: {th.__version__}\ncuda available: {th.cuda.is_available()}')

device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)