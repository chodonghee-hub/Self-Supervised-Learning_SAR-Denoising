import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=7):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        # layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.MaxPool2d(3, 3, 3))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out

class work_DnCNN(nn.Module) : 
    def __init__(self, channels, num_of_layers = 7) : 
        super(work_DnCNN, self).__init__()
        kernel_size = 11
        padding = 1
        features = 256
        # features = 64
        self.num_of_layers = num_of_layers
        r'''
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.MaxPool2d(3, 3, 3))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        '''
        self.start_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

        self.hidden_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=features, 
                out_channels=features,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            
            
            nn.MaxPool2d(
                kernel_size=11,
                stride=2,
            ),            
            
            # nn.AvgPool2d(
                # kernel_size=5,
                # stride=1,
            # ),
            
            nn.Dropout()
        )

        self.finish_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=features,
                out_channels=channels, 
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            )
        )


    def forward(self, seq):
        seq = self.start_layer(seq)
        for _ in range(self.num_of_layers - 2) :
            seq = self.hidden_layer(seq)
        seq = self.finish_layer(seq)
        return seq
