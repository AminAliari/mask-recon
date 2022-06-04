import torch.nn as nn


class MaskCNN(nn.Module):
    def __calc_output_size(self, img_dim, kernel, stride=1, padding=0):
        return int(((img_dim - kernel + 2 * padding) / stride) + 1)

    def __init__(self, label_count, img_dim, base_filter_size=12):
        super(MaskCNN, self).__init__()
        filter_sizes = [1 * base_filter_size, 2 * base_filter_size,
                        4 * base_filter_size, 6 * base_filter_size,
                        8 * base_filter_size, 10 * base_filter_size]

        output_size = self.__calc_output_size(img_dim[0], 5, 2, 2)
        output_size = self.__calc_output_size(output_size, 5, 2, 2)
        output_size = self.__calc_output_size(output_size, 3, 2, 1)
        output_size = self.__calc_output_size(output_size, 3, 2, 1)
        output_size = self.__calc_output_size(output_size, 3, 2, 1)
        output_size = self.__calc_output_size(output_size, 3, 2, 1)

        final_layer_size = filter_sizes[-1] * output_size * output_size

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=filter_sizes[0], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(filter_sizes[0]),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=filter_sizes[0], out_channels=filter_sizes[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(filter_sizes[1]),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=filter_sizes[1], out_channels=filter_sizes[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filter_sizes[2]),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=filter_sizes[2], out_channels=filter_sizes[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filter_sizes[3]),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=filter_sizes[3], out_channels=filter_sizes[4], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filter_sizes[4]),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=filter_sizes[4], out_channels=filter_sizes[5], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(filter_sizes[5]),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(final_layer_size, 1024),
            nn.ReLU(),

            nn.Linear(1024, label_count),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x
