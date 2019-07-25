import torch
import torch.nn as nn

from models.UNet import UNet

class StackUNet(nn.Module):
    r""" The Stack UNet class. The model is a sequence of UNet blocks.
    """
    
    def __init__(self, in_channels, num_classes, num_blocks, conv_type='double', residual=False, depth=4, 
                 activation='relu', dilation=1, upsample_type='upsample', **kwargs):
        r"""
        
        Parameters:
        -----------
        
        in_channels: int
            The number of channels in the input image.
            
        num_classes: int
            The number of channels in the output mask. Each channel corresponds to one of the classes and contains
            a mask of probabilities for image pixels to belong to this class.
        
        num_blocks: int 
            The number of UNet blocks in the model. Must be bigger then 1.
            
        conv_type: 'single', 'double' or 'triple' (default 'double')
            Defines the number of convolutions and activations in the model's blocks. If it is 'single', there 
            are one convolutional layer with kernel_size=3, padding=1, dilation=1, followed by activation. If 
            it is 'double' or 'triple', it is once or twice complemented by convolutional layer with kernel_size=3 
            and choosen dilation with corresponding padding, followed by activation.
        
        residual: bool (default False)
            Defines if the model's convolutional blocks have residual connections.
        
        depth: int (default 4)
            Defines the depth of encoding-decoding part in UNet blocks. Must be bigger then 2.
        
        activation: 'relu', 'prelu' or 'leaky_relu' (default 'relu')
            Defines the type of the activation function in the model's convolutional blocks.
        
        dilation: int (default 1) or list
            The dilation for the model's blocks convolutional layers.
        
        upsample_type: 'upsample' or 'convtranspose'
            Defines the tipe of upsampling in the UNet blocks.
        
        channels_sequence: list
            The list of the number of out_channels for decoding part of the UNet blocks. The length of it must match the depth.
            Example: for depth=4, it can be [64, 128, 256, 512]
            If it is not set, it will be set automaticly as it discribed in the original UNet peper.
            
        Applying:
        ---------
        
        >>> model = StackUNet(3, 1, 3, activation='leaky_relu', depth=3, channels_sequence=[32, 64, 64], dilation=2)
        >>> input = torch.tensor((1, 3, 256, 256))
        >>> output = model(input)

        For getting model ditails use torchsummary:
        
        >>> from torchsummary import summary
        >>> model = StackUNet(3, 1, 3)
        >>> summary(model, input_size=(3, 256, 256))
        """
        super().__init__()
        
        # Check if all model parameters are set correctly.
        
        if num_blocks < 2:
            raise ValueError("The number of blocks is expected to be bigger then 1.")
        
        if conv_type not in ['single', 'double', 'triple']:
            raise ValueError("The type of convolution blocks is expected to be 'single', 'double' or 'triple'.")
        if conv_type == 'single' and residual == True:
            raise NotImplementedError("For 'single' convolution blocks tupe residual is not expected to be True.")
          
        if depth < 3:
            raise ValueError("The depth of encoding and decoding part of the model is expected to be bigger then 2.")

        if activation not in ['relu', 'prelu', 'leaky_relu']:
            raise ValueError("The activation for convolution blocks is expected to be 'relu', 'prelu' or 'leaky_relu'.")
        if isinstance(dilation, int):
            if dilation not in [1, 2, 3]:
                raise ValueError("The dilation for convolution blocks is expected to be 1, 2 or 3.")
        if upsample_type not in ['upsample', 'convtranspose']:
            raise ValueError("The upsample type is expected to be Upsampling or ConvTranspose.")
            
        if 'channels_sequence' in kwargs.keys():
            channels_sequence = kwargs['channels_sequence']
            if len(channels_sequence) != depth:
                raise ValueError("The length of sequence of amount of channels in decoder must match to the depth of decoding part of the model.")
            for val in channels_sequence:
                if not isinstance(val, int) or val < 1:
                    raise ValueError("The amount of channels must to be possitive integer.")
                    
            for i in range(1, depth):
                if channels_sequence[i] < channels_sequence[i-1]:
                    raise ValueError("The amount of channels is expected to increase.")
                    
        # Define the number of out_channels in convolutional blocks in encoding part of the model.
        
        else:
            channels_sequence = [32]
            for i in range(depth-1):
                if i < 1:
                    channels_sequence.append(channels_sequence[-1] * 2)
                else:
                    channels_sequence.append(channels_sequence[-1])        
      
        # Layers initialization
        
        self.num_blocks = num_blocks
        out_channels = channels_sequence[0]
        
        self.UNet_block = UNet(in_channels, out_channels,
                               conv_type=conv_type, residual=residual, depth=depth, 
                               activation=activation, dilation=dilation, is_block=True, 
                               upsample_type=upsample_type, channels_sequence=channels_sequence)

        self.middle_conv = nn.Conv2d(out_channels+in_channels, in_channels, kernel_size=3, padding=1)
        self.last_conv = nn.Conv2d(out_channels, num_classes, kernel_size=3, padding=1)
        
    def forward(self, x): 
        
        x_res = x.clone()
        
        for i in range(self.num_blocks-1):
            x = self.UNet_block(x)
            x = torch.cat([x, x_res], dim=1)
            x = self.middle_conv(x)
            
        x = self.UNet_block(x)
        x = self.last_conv(x)
        
        return x
        
        
        
        