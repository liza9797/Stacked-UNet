import torch
import torch.nn as nn

CONV_TYPE = {'single': 1, 'double': 2, 'triple': 3}

class ConvBlock(nn.Module):
    r""" The basic convolutional block with activations.
    """
    
    def __init__(self, in_channels, out_channels, conv_type='double', dilation=1, activation='relu', residual=False):
        r"""
        
        Parameters:
        -----------
        
        in_channels: int
            The number of channels in the input tensor.
            
        out_channels: int
            The number of channels in the output tensor.
            
        conv_type: 'single', 'double' or 'triple' (default 'double')
            Defines the number of convolutions and activations in the block. If it is 'single', there are one 
            convolutional layer with kernel_size=3, padding=1, dilation=1, followed by activation. If it is 
            'double' or 'triple', it is once or twice complemented by convolutional layer with kernel_size=3 
            and choosen dilation with corresponding padding, followed by activation.
            
        dilation: int (default 1) or list
            The dilations for the convolutional layers.
            
        activation: 'relu', 'prelu' or 'leaky_relu' (default 'relu')
            Defines the type of the activation function.
            
        residual: bool (default False)
            Defines if the block has residual connection.
        
        """

        super().__init__()
        
        self.num_convs = CONV_TYPE[conv_type]
        self.conv_layers = nn.ModuleList([])
        
        if isinstance(dilation, int):
            dilation = [dilation] * self.num_convs
        for i in range(self.num_convs):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=dilation[i], dilation=dilation[i]))
            in_channels = out_channels
        
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        if activation == 'prelu':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU(inplace=True)

        self.residual = residual
        
    def forward(self, x):
        for i in range(self.num_convs):
            if i == 1 and self.residual:
                x_out = x.clone()
            
            x = self.activation(self.conv_layers[i](x))

        if self.residual:
            return x_out + x
        else:
            return x

class UNet(nn.Module):
    r""" The basic UNet block. It can be used as completed model or as a part of the Stacked UNet model.
    """
    
    def __init__(self, in_channels, out_channels, conv_type='double', residual=False, depth=4, 
                 activation='relu', dilation=1, is_block=False, upsample_type='upsample', **kwargs):
        r"""
        
        Parameters:
        -----------
        
        in_channels: int
            The number of channels in the input image.
            
        out_channels: int
            The number of channels in the output mask. Each channel corresponds to one of the classes and contains
            a mask of probabilities for image pixels to belong to this class.
            
        conv_type: 'single', 'double' or 'triple' (default 'double')
            Defines the number of convolutions and activations in the model's blocks. If it is 'single', there 
            are one convolutional layer with kernel_size=3, padding=1, dilation=1, followed by activation. If 
            it is 'double' or 'triple', it is once or twice complemented by convolutional layer with kernel_size=3 
            and choosen dilation with corresponding padding, followed by activation.
        
        residual: bool (default False)
            Defines if the model's convolutional blocks have residual connections.
        
        depth: int (default 4)
            Defines the depth of encoding-decoding part. Must be bigger then 2.
        
        activation: 'relu', 'prelu' or 'leaky_relu' (default 'relu')
            Defines the type of the activation function in the model's convolutional blocks.
        
        dilation: int (default 1) or list
            The dilation for the model's blocks convolutional layers.
            
        is_block: bool (default False)
            It should be set to True if the model is used as a block in Stacked UNet model. 
            If it is True:
                * the deepest convolutional block has the same number of channels, as the privious one;
                * there is no last convolutional layer, so, the model ends with activation layer from convolutional 
                  block.
            If it is False:
                * the deepest layer has two times more out channels than previous one;
                * the model ends with additional convolutional layer.
        
        upsample_type: 'upsample' or 'convtranspose'
            Defines the tipe of upsampling in the model.
        
        channels_sequence: list
            The list of the number of out_channels for decoding part of the model. The length of it must match the depth.
            Example: for depth=4, it can be [64, 128, 256, 512]
            If it is not seted, it will be setted automaticly as it discribed in the original UNet peper.
            
        Applying:
        ---------
        
        >>> model = UNet(3, 1, activation='leaky_relu', depth=3, channels_sequence=[32, 64, 64], dilation=2)
        >>> input = torch.tensor((1, 3, 256, 256))
        >>> output = model(input)

        For getting model ditails use torchsummary:
        
        >>> from torchsummary import summary
        >>> model = UNet(3, 1)
        >>> summary(model, input_size=(3, 256, 256))
        """

        super().__init__()
        
        # Check if all model parameters are set correctly.
        
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
            channels_sequence = [64]
            for i in range(depth-1):
                if i < 3:
                    channels_sequence.append(channels_sequence[-1] * 2)
                else:
                    channels_sequence.append(channels_sequence[-1])
        
        # Define the number of out_channels in convolutional blocks in decoding part of the model.
        inverse_channels_sequence = channels_sequence[::-1]
        if is_block:
            inverse_channels_sequence.insert(0, channels_sequence[-1] )
        else:
            inverse_channels_sequence.insert(0, channels_sequence[-1] * 2)
        channels_sequence.insert(0, in_channels)
   
        # Layers initialization
        
        self.is_block = is_block
        self.conv_down = nn.ModuleList([])
        self.conv_up = nn.ModuleList([])
        
        for i in range(1, depth+1):

            self.conv_down.append(ConvBlock(channels_sequence[i-1], channels_sequence[i], conv_type=conv_type,
                                            residual=residual, activation=activation, dilation=dilation))
            
            self.conv_up.append(ConvBlock(inverse_channels_sequence[i-1]+channels_sequence[-i], 
                                                  inverse_channels_sequence[i], conv_type=conv_type, residual=residual, 
                                                  activation=activation, dilation=dilation))
        
        self.conv_middle = ConvBlock(channels_sequence[-1], inverse_channels_sequence[0], conv_type=conv_type, 
                                     residual=residual, activation=activation, dilation=dilation)
        if not is_block:
            self.conv_last = nn.Conv2d(inverse_channels_sequence[-1], 
                                                      out_channels, 3, padding=1)
     
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  

    def forward(self, x):
        
        encoding = []
        for conv_block in self.conv_down:
            x = conv_block(x)
            encoding.append(x)
            x = self.maxpool(x)
         
        x = self.conv_middle(x)
        
        for i, conv_block in enumerate(self.conv_up):
            x = self.upsample(x)
            x = torch.cat([x, encoding[::-1][i]], dim=1)
            x = conv_block(x)
            
        if not self.is_block:
            x = self.conv_last(x)
   
        return x
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
            
            
            
            