import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import TransformerEncoderLayer
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torchvision
import matplotlib.pyplot as plt
from torch import optim, nn
from tqdm import tqdm
from torchsummary import summary

dataset_classes = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


class Parameters:

    def __init__(self):
        #Image
        self.img_size = 32
        self.n_channels = 3
        self.patch_size = 8
        self.n_classes = 100

        self.n_patch = self.img_size * self.img_size // self.patch_size ** 2
        self.n_embds = self.patch_size ** 2 * self.n_channels

        #Traing
        self.learning_rate = 0.005
        self.epochs = 50
        self.dropout = 0.2
        self.optimizer = optim.AdamW
        self.activation = 'gelu'
        self.criterion = nn.CrossEntropyLoss
        self.save_every_epoch = 4
        self.save_PATH = './model_save_stage_CIFAR100/'
        self.load_PATH = './model_save_stage_CIFAR100/'

        #Dataset
        self.train_batch_size = 1024
        self.test_batch_size = 128
        self.n_workers = 1
        #Transformer
        self.n_heads = 8
        self.n_layers = 8
        self.dim_feedforward = 2048
        self.dropout_transformer = 0.2


class VisualTransformer(nn.Module):

    def __init__(self):
        super(VisualTransformer, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(self.device)

        self.test_loader = None
        self.train_loader = None

        self.patching = (nn.Sequential
            (
            nn.Conv2d(Parameters().n_channels, Parameters().n_embds, Parameters().patch_size, Parameters().patch_size),
            nn.GELU(),
            nn.AvgPool2d(2),
            nn.Flatten(2),
            nn.Dropout(Parameters().dropout)

        ))

        self.class_token = nn.Parameter(torch.rand(1, Parameters().n_channels, Parameters().n_embds),
                                        requires_grad=True)

        self.position_token = nn.Parameter(
            torch.rand(1, Parameters().n_patch + Parameters().n_channels, Parameters().n_embds),
            requires_grad=True)

        self.EncoderLayer = TransformerEncoderLayer(Parameters().n_embds,
                                                    Parameters().n_heads,
                                                    dim_feedforward=Parameters().dim_feedforward,
                                                    dropout=Parameters().dropout_transformer,
                                                    activation=Parameters().activation,
                                                    batch_first=True,
                                                    norm_first=True)

        self.Encoder = nn.TransformerEncoder(self.EncoderLayer, num_layers=Parameters().n_layers)

        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=Parameters().n_embds),
            nn.Linear(Parameters().n_embds, Parameters().n_classes),
            nn.GELU(),
            nn.Linear(Parameters().n_classes, Parameters().n_classes),
            nn.GELU(),
            nn.Linear(Parameters().n_classes, Parameters().n_classes),
            nn.GELU(),
            nn.Linear(Parameters().n_classes, Parameters().n_classes)
            ,
            nn.GELU(),
            nn.Linear(Parameters().n_classes, Parameters().n_classes),
            nn.GELU(),
            nn.Linear(Parameters().n_classes, Parameters().n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract patches from images
        #[B,C,X,Y]
        class_token = self.class_token.expand(x.shape[0], -1, -1)  # Adding batch dimension
        position_token = self.position_token.expand(x.shape[0], -1, -1)  # Adding batch dimension
        x = self.patching(x).permute(0, 2, 1)

        x = torch.cat((x, class_token), dim=1)
        x = torch.cat((x, position_token), dim=1)

        x = self.Encoder(x)
        x = self.MLP(x[:, 0, :])
        return x

    def save_model(self, epoch):

        epoch = epoch + 1
        if epoch == Parameters().epochs:
            tqdm.write('Saving model...')
            torch.save(super().state_dict(), Parameters().save_PATH + 'ViT_model_last_epoch' + '.pth')

        else:
            if epoch % Parameters().save_every_epoch == 0:
                tqdm.write('Saving model...')
                torch.save(super().state_dict(), Parameters().save_PATH + 'ViT_model_epoch_' + str(epoch) + '.pth')

    def load_model(self, PATH):
        super().load_state_dict(torch.load(PATH))

    def load_MNIST(self):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_dataset = torchvision.datasets.MNIST(root='./MNIST/train', train=True, download=True,
                                                   transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./MNIST/test', train=False, download=True,
                                                  transform=transform)

        self.train_loader = DataLoader(train_dataset, batch_size=Parameters().train_batch_size, shuffle=True,
                                       generator=torch.Generator(device='cuda'))
        self.test_loader = DataLoader(test_dataset, batch_size=Parameters().test_batch_size, shuffle=False,
                                      generator=torch.Generator(device='cuda'))

    def load_any_dataset(self, dataset: torchvision.datasets = torchvision.datasets.CIFAR10,
                         save_dataset_path='./CIFAR10/'):
        self.train_loader = DataLoader(
            dataset=dataset(root=save_dataset_path + 'train', train=True,
                            download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                            ]))
            ,
            drop_last=True,
            batch_size=Parameters().train_batch_size,
            pin_memory=True,
            num_workers=Parameters().n_workers,
            generator=torch.Generator(device='cuda')
        )

        self.test_loader = DataLoader(dataset=dataset(root=save_dataset_path + 'test', train=False,
                                                      download=True,
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor()
                                                      ])), batch_size=Parameters().test_batch_size)

    def start_train(self, save_model=True):
        criterion = Parameters().criterion()
        optimizer = Parameters().optimizer(super().parameters(), lr=Parameters().learning_rate)

        if self.train_loader is None or self.test_loader is None:
            raise ValueError("Train and test loaders have not been set")

        super().train()
        for epoch in range(Parameters().epochs):
            running_loss = 0.0
            with tqdm(self.train_loader) as t:
                for images_in, labels_in in t:
                    images_in = images_in.to(self.device)
                    labels_in = labels_in.to(self.device)

                    optimizer.zero_grad()

                    images_out = self.forward(images_in)
                    loss = criterion(images_out, labels_in)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images_in.size(0)  # multipy by batch size

                epoch_loss = running_loss / len(self.train_loader.dataset)

                if save_model:
                    self.save_model(epoch)
                tqdm.write(f"\nEpoch [{epoch + 1}/{Parameters().epochs}], Loss: {epoch_loss:.4f}")

    def testing(self, offset=0, show_images_grid=(2, 3), include='ACC-VIS'):

        accuracy = 0
        labels_out = []
        images_in = []
        labels_in = []

        for _images_in, _labels_in in self.test_loader:
            _images_in = _images_in.to(self.device)

            label_out = self.forward(_images_in)
            y_pred = torch.argmax(label_out, dim=1)

            images_in.append(_images_in)

            labels_in.append(_labels_in.cpu().numpy())

            labels_out.append(y_pred.cpu().numpy())

            accuracy += accuracy_score(_labels_in.to('cpu'), y_pred.to('cpu'))

        print('Accuracy:', 1 - accuracy / len(self.test_loader.dataset))
        print(torch.max(images_in[0]), torch.min(images_in[0]))
        super().eval()

        counter = 0

        batch = min(offset, len(images_in))
        with torch.no_grad():
            plt.figure()
            fig, ax = plt.subplots(show_images_grid[0], show_images_grid[1])
            for i in range(show_images_grid[0]):
                for j in range(show_images_grid[1]):

                    ax[i][j].imshow(images_in[batch].cpu()[counter].permute(1, 2, 0))
                    text = str(labels_out[batch][counter]) + ' True:' + str(labels_in[batch][counter])
                    ax[i][j].set_title(text)
                    if counter + 1 == Parameters().test_batch_size:
                        counter = 0
                        batch = min(batch + 1, len(images_in))
                    else:
                        counter += 1

            plt.show()

    def load_single_image(self, img_path='./boat.png'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))

        ])
        img = torchvision.io.image.read_image(img_path, torchvision.io.ImageReadMode.RGB)
        img = torchvision.transforms.ToPILImage()(img)
        img: torch.Tensor = transform(img)
        img = img.to('cuda')
        img = img.unsqueeze(0)
        label_out = model(img)

        label_out = torch.argmax(label_out, dim=1)
        return dataset_classes[label_out.cpu().numpy()[0]]


if __name__ == '__main__':
    model = VisualTransformer()
    model.load_any_dataset(torchvision.datasets.CIFAR100, save_dataset_path='./CIFAR100/')
    summary(model, (3, 32, 32))
    model.start_train(False)
    #model.testing()
