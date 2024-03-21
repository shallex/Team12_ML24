import torch
from tqdm import tqdm
from IPython.display import clear_output
import random
from misc.utils import get_outputs
import numpy as np


class MNIST_ConvEncoder(torch.nn.Module):
    """
    MNIST Convolutional encoder.
    
    Parameters
    ----------
    latent_dim : int
        Latent representation dimension.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Noise.
        self.dropout = torch.nn.Dropout(0.1)
        
        # Activations.
        #self.activation = torch.nn.LeakyReLU()
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        # Convolutions.
        self.conv2d_1 = torch.nn.Conv2d(1, 8, kernel_size=3, padding='same')
        self.conv2d_2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding='same')
        self.conv2d_3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding='same')
        
        self.maxpool2d = torch.nn.MaxPool2d((2,2))
        
        # Dense.
        self.linear_1 = torch.nn.Linear(288, 128)
        self.linear_2 = torch.nn.Linear(128, self.latent_dim)
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Convolution №1
        x = self.dropout(x)
        x = self.conv2d_1(x)
        x = self.maxpool2d(x)
        layer_1 = self.activation(x)
        
        # Convolution №2
        x = self.dropout(layer_1)
        x = self.conv2d_2(x)
        x = self.maxpool2d(x)
        layer_2 = self.activation(x)
        
        # Convolution №3
        x = self.dropout(layer_2)
        x = self.conv2d_3(x)
        x = self.maxpool2d(x)
        layer_3 = self.activation(x)
        
        # Dense №1
        x = torch.flatten(layer_3, 1)
        x = self.linear_1(x)
        layer_4 = self.activation(x)
        
        # Dense №2
        x = self.linear_2(layer_4)
        layer_5 = self.sigmoid(x)
        
        return layer_5



class MNIST_ConvDecoder(torch.nn.Module):
    """
    MNIST Convolutional decoder.
    
    Parameters
    ----------
    latent_dim : int
        Latent representation dimension.
    """
        
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Activations.
        #self.activation = torch.nn.LeakyReLU()
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        # Convolutions.
        self.conv2d_1 = torch.nn.Conv2d(32, 16, kernel_size=3, padding='same')
        self.conv2d_2 = torch.nn.Conv2d(16, 8, kernel_size=3, padding='same')
        self.conv2d_3 = torch.nn.Conv2d(8, 1, kernel_size=3, padding='same')
        
        self.upsample = torch.nn.Upsample(scale_factor=2)
        
        # Dense.
        self.linear_1 = torch.nn.Linear(latent_dim, 128)
        self.linear_2 = torch.nn.Linear(128, 1568)
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Dense №1
        x = self.linear_1(x)
        layer_1 = self.activation(x)
        
        # Dense №2
        x = self.linear_2(layer_1)
        layer_2 = self.activation(x)
        
        # Convolution №1
        x = torch.reshape(layer_2, (-1, 32, 7, 7))
        x = self.conv2d_1(x)
        x = self.upsample(x)
        layer_3 = self.activation(x)
        
        # Convolution №2
        x = self.conv2d_2(layer_3)
        x = self.upsample(x)
        layer_4 = self.activation(x)
        
        # Convolution №3
        x = self.conv2d_3(layer_4)
        layer_5 = x #self.sigmoid(x)
        
        return layer_5
    
    

class CIFAR10_ConvEncoder(torch.nn.Module):
    """
    CIFAR10 Convolutional encoder.
    
    Parameters
    ----------
    latent_dim : int
        Latent representation dimension.
    """
    
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Noise.
        self.dropout = torch.nn.Dropout(0.1)
        
        # Activations.
        #self.activation = torch.nn.LeakyReLU()
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Tanh()
        
        # Convolutions.
        self.conv2d_1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding='same')
        self.conv2d_2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding='same')
        self.conv2d_3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding='same')
        
        self.maxpool2d = torch.nn.MaxPool2d((2,2))
        
        # Dense.
        self.linear_1 = torch.nn.Linear(64*4*4, 128)
        self.linear_2 = torch.nn.Linear(128, self.latent_dim)
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Convolution №1
        x = self.dropout(x)
        x = self.conv2d_1(x)
        x = self.maxpool2d(x)
        layer_1 = self.activation(x)
        
        # Convolution №2
        x = self.dropout(layer_1)
        x = self.conv2d_2(x)
        x = self.maxpool2d(x)
        layer_2 = self.activation(x)
        
        # Convolution №3
        x = self.dropout(layer_2)
        x = self.conv2d_3(x)
        x = self.maxpool2d(x)
        layer_3 = self.activation(x)
        
        # Dense №1
        x = torch.flatten(layer_3, 1)
        x = self.linear_1(x)
        layer_4 = self.activation(x)
        
        # Dense №2
        x = self.linear_2(layer_4)
        layer_5 = self.sigmoid(x)
        
        return layer_5



class CIFAR10_ConvDecoder(torch.nn.Module):
    """
    CIFAR10 Convolutional decoder.
    
    Parameters
    ----------
    latent_dim : int
        Latent representation dimension.
    """
        
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Activations.
        #self.activation = torch.nn.LeakyReLU()
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
        # Convolutions.
        self.conv2d_1 = torch.nn.Conv2d(64, 32, kernel_size=3, padding='same', padding_mode='reflect')
        self.conv2d_2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding='same', padding_mode='reflect')
        self.conv2d_3 = torch.nn.Conv2d(32, 16, kernel_size=3, padding='same', padding_mode='reflect')
        self.conv2d_4 = torch.nn.Conv2d(16, 3, kernel_size=3, padding='same', padding_mode='reflect')
        
        self.upsample = torch.nn.Upsample(scale_factor=2)
        
        # Dense.
        self.linear_1 = torch.nn.Linear(latent_dim, 128)
        self.linear_2 = torch.nn.Linear(128, 64*4*4)
        
        
    def forward(self, x: torch.tensor) -> torch.tensor:
        # Dense №1
        x = self.linear_1(x)
        layer_1 = self.activation(x)
        
        # Dense №2
        x = self.linear_2(layer_1)
        layer_2 = self.activation(x)
        
        # Convolution №1
        x = torch.reshape(layer_2, (-1, 64, 4, 4))
        x = self.conv2d_1(x)
        x = self.upsample(x)
        layer_3 = self.activation(x)
        
        # Convolution №2
        x = self.conv2d_2(layer_3)
        x = self.upsample(x)
        layer_4 = self.activation(x)
        
        # Convolution №3
        x = self.conv2d_3(layer_4)
        x = self.upsample(x)
        layer_5 = self.activation(x)
        
        # Convolution №4
        x = self.conv2d_4(layer_5)
        layer_6 = x
        
        return layer_6
    
    
class Autoencoder(torch.nn.Module):
    """
    Autoencoder.
    
    Parameters
    ----------
    encoder : torch.nn.Module
        Encoder.
    decoder : torch.nn.Module
        Decoder.
    latent_dim : int
        Latent representation dimension.
    sigma : float
        Standard deviation of additive Gaussian noise,
        injected into the latent representation.
    """

    def __init__(self, encoder, decoder, sigma: float=0.1):
        super().__init__()
        
        # Encoder and decoder.
        self.encoder = encoder
        self.decoder = decoder
             
    def forward(self, x: torch.tensor) -> torch.tensor:
        latent = self.encoder(x)
        
        return self.decoder(latent)
    
    
    def encode(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)
    
    
    def decode(self, x: torch.tensor) -> torch.tensor:
        return self.decoder(x)


def evaluate_model(model, dataloader, loss, device) -> float:
    # Exit training mode.
    was_in_training = model.training
    model.eval()
  
    with torch.no_grad():
        avg_loss = 0.0
        total_samples = 0
        for batch in dataloader:
            x, y = batch
            batch_size = x.shape[0]
            
            y_pred = model(x.to(device))
            _loss = loss(y_pred, y.to(device))

            avg_loss += _loss.item() * batch_size
            total_samples += batch_size
            
        avg_loss /= total_samples
        
    # Return to the original mode.
    model.train(was_in_training)
    
    return avg_loss

def evaluate_model_samplewise(model, dataloader, loss, device) -> float:
    # Exit training mode.
    was_in_training = model.training
    model.eval()
    
    out_list = []
  
    with torch.no_grad():
        avg_loss = 0.0
        total_samples = 0
        for batch in dataloader:
            x, y = batch
            batch_size = x.shape[0]
            
            y_pred = model(x.to(device))
            _loss = loss(y_pred, y.to(device))
            _loss = torch.mean(_loss, dim=list(range( len(_loss.shape) ))[1:])
            out_list.append(_loss.cpu().numpy())

        
    # Return to the original mode.
    model.train(was_in_training)
    
    return np.hstack(out_list)



def train_autoencoder(
    autoencoder, 
    train_dataloader, 
    test_dataloader,
    autoencoder_loss,
    autoencoder_opt,
    device, 
    n_epochs: int=10, 
    train_dataloader_nonshuffle=None,
    loss_samplewise=None,
    callback: callable=None, ) -> dict():

    
    autoencoder_metrics = {
        "train_loss" : [],
        "test_loss" : [],
    }
    
    samplewise_metrics = {}
    
    
    if train_dataloader_nonshuffle is not None:
        bn_logits = []
        
        # samplewise metrics
        if loss_samplewise is not None:
            samplewise_metrics['samplewise_loss'] = []
    
    for epoch in tqdm(range(1, n_epochs + 1)):
        print(f"Epoch №{epoch}")
        
        sum_loss = 0.0
        total_samples = 0
        for index, batch in (enumerate(train_dataloader)):
            x, y = batch
            batch_size = x.shape[0]
            
            autoencoder_opt.zero_grad()
            y_pred = autoencoder(x.to(device))
            _loss = autoencoder_loss(y_pred, y.to(device))
            _loss.backward()
            autoencoder_opt.step()
            
            sum_loss += _loss.item() * len(batch)
            total_samples += len(batch)
        
        autoencoder_metrics["train_loss"].append(sum_loss / total_samples)
        
        # calculating sample-wise metrics
        if train_dataloader_nonshuffle is not None:
            bn_logits.append(get_outputs(autoencoder.encoder, train_dataloader_nonshuffle, device).numpy())
            
            # Samplewise loss
            if loss_samplewise is not None:
                samplewise_metrics['samplewise_loss'].append(evaluate_model_samplewise(autoencoder, train_dataloader_nonshuffle, loss_samplewise, device))
            
        
        #train_loss = evaluate_model(autoencoder, train_dataloader, autoencoder_loss, device)
        #autoencoder_metrics["train_loss"].append(train_loss)
        test_loss = evaluate_model(autoencoder, test_dataloader, autoencoder_loss, device)
        autoencoder_metrics["test_loss"].append(test_loss)
        
        if not (callback is None):
            callback(autoencoder, autoencoder_metrics)
            
    if train_dataloader_nonshuffle is not None:
        return {"metrics": autoencoder_metrics, "bn_logits": bn_logits, "samplewise_metrics": samplewise_metrics}
    else:
        return {"metrics": autoencoder_metrics}