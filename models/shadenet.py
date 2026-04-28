import torch
import torch.nn as nn 
from .modules import Down, Attention_block, Decoder_Up, DoubleConv, OutConv, SelfAttention

class ShadeNet(nn.Module):
    def __init__(self, mid_layers, n_classes = 1):
        super().__init__()

        assert len(mid_layers) == 5, "ShadeNet only has 4 encoders; len(mid_layers) == 4"
        

        # Decoder layers
        self.encoder1 = DoubleConv(3, mid_layers[0])
        self.encoder2 = Down(mid_layers[0], mid_layers[1])
        self.encoder3 = Down(mid_layers[1], mid_layers[2])
        self.encoder4 = Down(mid_layers[2], mid_layers[3])
        self.bottleneck = Down(mid_layers[3], mid_layers[4])

        
        # Attention Layer and decoder Layers
        self.decoder4 = Decoder_Up(mid_layers[4], mid_layers[3]) # code block 
        self.attenion4 = SelfAttention(mid_layers[3])
        self.up_conv4 = DoubleConv(2 * mid_layers[3], mid_layers[3])

        self.decoder3 = Decoder_Up(mid_layers[3], mid_layers[2])
        self.attenion3 = Attention_block(mid_layers[2])
        self.up_conv3 = DoubleConv(2 * mid_layers[2], mid_layers[2])

        self.decoder2 = Decoder_Up(mid_layers[2], mid_layers[1])
        self.attenion2 = Attention_block(mid_layers[1])
        self.up_conv2 = DoubleConv(2 * mid_layers[1], mid_layers[2])

        self.decoder1 = Decoder_Up(mid_layers[1], mid_layers[0])
        self.attention1 = Attention_block(mid_layers[0])
        self.up_conv1 = DoubleConv(2 * mid_layers[0], mid_layers[0])

        # outconv 
        # for segmentation task, predicting image mask 
        self.maskOutConv = OutConv(mid_layers[0], n_classes)

        # for image reconstruction 
        self.reconstructionOutConv = OutConv(mid_layers[0], 3)



    def forward(self, x):
        

        # decoder section 
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.bottleneck(x4)


        # encoder + attention 
        # bottleneck layer 
        dec_out = self.decoder4(x5)
        v4 = self.attenion4(x4)
        dec_out = torch.cat((dec_out,v4), dim = 1)
        dec_out = self.up_conv4(dec_out)

        # Level 3
        dec_out = self.decoder3(dec_out)
        v3 = self.attention3(dec_out, x3)

        dec_out = torch.cat((dec_out, v3), dim=1)
        dec_out = self.up_conv3(dec_out)

        # Level 2
        dec_out = self.decoder2(dec_out)
        v2 = self.attention2(dec_out, x2)

        dec_out = torch.cat((dec_out, v2), dim=1)
        dec_out = self.up_conv2(dec_out)


        # Level 1
        dec_out = self.decoder1(dec_out)
        v1 = self.attention1(dec_out, x1)

        dec_out = torch.cat((dec_out, v1), dim=1)
        dec_out = self.up_conv1(dec_out)    


        logits = self.maskOutConv(dec_out)
        reconstructed = self.reconstructionOutConv(dec_out)

        return logits, reconstructed



        
        


