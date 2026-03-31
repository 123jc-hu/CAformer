import torch
from torch import nn
from torchsummary import summary
# from main import parser


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.n_channels = configs.n_channels
        self.fs = configs.fs
        self.n_classes = configs.n_class
        self.dropout_rate = configs.dropout_rate
        self.kernel_length_list1 = torch.tensor([int(self.fs // i) for i in [2, 4, 8]])
        self.kernel_length_list2 = self.kernel_length_list1 // 4

        self.Inception1_branch1 = nn.Sequential(
            nn.Conv2d(1, 8, (1, self.kernel_length_list1[0]), stride=1, bias=False, padding=(0, self.kernel_length_list1[0] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(8, 16, (self.n_channels, 1), stride=1, bias=False, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.Inception1_branch2 = nn.Sequential(
            nn.Conv2d(1, 8, (1, self.kernel_length_list1[1]), stride=1, bias=False, padding=(0, self.kernel_length_list1[1] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(8, 16, (self.n_channels, 1), stride=1, bias=False, groups=8,),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.Inception1_branch3 = nn.Sequential(
            nn.Conv2d(1, 8, (1, self.kernel_length_list1[2]), stride=1, bias=False, padding=(0, self.kernel_length_list1[2] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Conv2d(8, 16, (self.n_channels, 1), stride=1, bias=False, groups=8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.Inception2_branch1 = nn.Sequential(
            nn.Conv2d(48, 8, (1, self.kernel_length_list2[0]), stride=1, bias=False, padding=(0, self.kernel_length_list2[0] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.Inception2_branch2 = nn.Sequential(
            nn.Conv2d(48, 8, (1, self.kernel_length_list2[1]), stride=1, bias=False, padding=(0, self.kernel_length_list2[1] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.Inception2_branch3 = nn.Sequential(
            nn.Conv2d(48, 8, (1, self.kernel_length_list2[2]), stride=1, bias=False, padding=(0, self.kernel_length_list2[2] // 2)),
            nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),
        )
        self.output_module = nn.Sequential(
            nn.AvgPool2d((1, 2), stride=(1, 2)),

            nn.Conv2d(24, 12, (1, 8), stride=1, bias=False, padding=(0, 4)),
            nn.BatchNorm2d(12),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),

            nn.AvgPool2d((1, 2), stride=(1, 2)),

            nn.Conv2d(12, 6, (1, 4), stride=1, bias=False, padding=(0, 2)),
            nn.BatchNorm2d(6),
            nn.ELU(),
            nn.Dropout(p=self.dropout_rate),

            nn.AvgPool2d((1, 2), stride=(1, 2)),

            nn.Flatten(),
            nn.Linear(24, self.n_classes, bias=False),
            # nn.Softmax(dim=1)
        )

        self.AvgPool2d = nn.AvgPool2d((1, 4), stride=(1, 4))

    def forward(self, x):
        x1 = self.Inception1_branch1(x)  # (batch, 16, 1, 129)
        x2 = self.Inception1_branch2(x)
        x3 = self.Inception1_branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)  # (batch, 48, 1, 129)
        x = self.AvgPool2d(x)
        x1 = self.Inception2_branch1(x)  # (batch, 8, 1, 33)
        x2 = self.Inception2_branch2(x)
        x3 = self.Inception2_branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)  # (batch, 24, 1, 33)
        x = self.output_module(x)

        return x, None


# if __name__ == '__main__':
    # args = parser.parse_args()
    # net = Model(args)
    # print(summary(net, (1, 62, 128)))
