
import torch.nn as nn



# define the neural network
class WBNET(nn.Module):
  def __init__(self):
    super().__init__()

    self.layer1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=2, padding=0),
      nn.ReLU(),
      nn.BatchNorm2d(num_features=6),
      nn.MaxPool2d(kernel_size=2),
    )

    self.layer2 = nn.Sequential(
      nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=2, padding=0),
      nn.ReLU(),
      nn.BatchNorm2d(num_features=16),
      nn.MaxPool2d(kernel_size=2)
    )

    self.layer3 = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features=16 * 63 * 63, out_features=120),
      nn.ReLU(),
      nn.Linear(in_features=120, out_features=84),
      nn.ReLU(),
      nn.Linear(in_features=84, out_features=2),
      nn.Softmax(dim=0)
    )

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = x.reshape(x.size(0), -1)
    x = self.layer3(x)
    return x

  def predict(self,x):
    if self.forward(x)>0.5:
      return 1
    else:
      return 0




