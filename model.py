import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch


# Kernel size 5, Strides 2 and padding 0 is used
# As the model has only two layers for 28x28 image, kernel size is kept as moderate such as 5 in this experiments
# It is a digit classification problem which requires to learn only generic features rather than intricate or detailed one.
# Hence filter size is 5 and strides are 2 where, model tolerates the loss of exact image.
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 10)

    # Own softmax function
    def soft_max(self, x):
        means = torch.mean(x, 1, keepdim=True)[0]
        x_exp = torch.exp(x-means)
        x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
        return x_exp/x_exp_sum

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x) 
        # return self.soft_max(x)