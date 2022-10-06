from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        initial_filters = 8
        self.network = nn.Sequential(

                nn.Conv2d(1, initial_filters, 3, padding="same"),
                nn.BatchNorm2d(initial_filters),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(initial_filters, initial_filters*2, 3, padding="same"),
                nn.BatchNorm2d(initial_filters*2),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(initial_filters*2, initial_filters*4, 3, padding="same"),
                nn.BatchNorm2d(initial_filters*4),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(initial_filters*4, initial_filters*8, 3, padding="same"),
                nn.BatchNorm2d(initial_filters*8),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(initial_filters*8, initial_filters*16, 3, padding="same"),
                nn.BatchNorm2d(initial_filters*16),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),
                #nn.Dropout(0.5),
                nn.Linear(2*1* initial_filters*16, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 25),
        )

    def forward(self, x):
        return self.network(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        initial_filters = 64
        self.encoder = nn.Sequential(
                nn.Conv2d(1, initial_filters, 3, padding="same"),
                nn.BatchNorm2d(initial_filters),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(initial_filters, initial_filters//2, 3, padding="same"),
                nn.BatchNorm2d(initial_filters//2),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(initial_filters//2, initial_filters//4, 3, padding="same"),
                nn.BatchNorm2d(initial_filters//4),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # for input shape of (initial_filters//4, 72, 32) output of this layer is (initial_filters//4, 9, 4)
        )

        self.decoder = nn.Sequential(
                nn.Conv2d(initial_filters//4, initial_filters//4, 3, padding="same"),
                nn.BatchNorm2d(initial_filters//4),
                nn.ReLU(),
                nn.ConvTranspose2d(initial_filters//4, initial_filters//2, 2, stride=2),

                nn.Conv2d(initial_filters//2, initial_filters//2, 3, padding="same"),
                nn.BatchNorm2d(initial_filters//2),
                nn.ReLU(),
                nn.ConvTranspose2d(initial_filters//2, initial_filters, 2, stride=2),

                nn.Conv2d(initial_filters, initial_filters, 3, padding="same"),
                nn.BatchNorm2d(initial_filters),
                nn.ReLU(),
                nn.ConvTranspose2d(initial_filters, 1, 2, stride=2),
        )

        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(initial_filters//4 * 9 * 4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 25),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        pars = self.classifier(encoded)
        return decoded, pars
