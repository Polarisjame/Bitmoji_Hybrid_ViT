import numpy as np
import torch
from tqdm import tqdm

from models import VIT
from train import args
from data.dataloader_lightning import VITSet
import pandas as pd

model = VIT(args)
state_dict = torch.load(r'./checkpoint/model_epoch4.pth')
model.load_state_dict(state_dict)
data = VITSet()
test_loader = data.test_dataloader()
device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"

with tqdm(total=len(test_loader)) as t:
    result = []
    for (data_x, _) in test_loader:
        # print(data_y)
        data_x = data_x.to(device)
        out = model(data_x)
        result.extend(out.data.cpu().numpy())

result = [np.array(1) if a == 1 else -1 for a in result]
ind = list(range(3000, 4084))
df = pd.DataFrame(columns=['image_id','is_males'],data=[ind,result])
