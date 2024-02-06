import numpy as np
import torch

class HemaModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(HemaModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        n_skip = encoder.n_blocks
        self.layers_to_get = np.arange(1, n_skip-1, n_skip/5).round().astype("int")

        print(f"Using layers {self.layers_to_get} to get UNet skip connections.")

    def forward(self, x):
        encoder_output = self.encoder.get_intermediate_layers(x,
            n=self.layers_to_get, return_class_token=False) # last index is last layer output, etc.
        
        output = self.decoder(encoder_output)
        return output