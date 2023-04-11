# Let's create a PatchEmbedding class that will take an image of various size and output a tensor of patches with varying number of patches. The only important parameter is the size of the patches
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """Turns a 2D input image of variable size into a learnable 1D sequence of patches.
    
    Args:
        patch_size (int): Size of the patches to be extracted from the image.
        emb_dim (int): Dimension of the embedding.
        max_length (int): Maximum length of the sequence.
    """ 
    def __init__(self, 
                 patch_size=10,
                 emb_dim=32,
                 max_length=256,
                 img_channels=1):
        
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.max_length = max_length
        self.linear = nn.Linear(patch_size*patch_size*img_channels, emb_dim)
    
    def to_patch(self,x):
        img_dim = x.shape[-3:]
        
        # Get the number of patches required to cover the image:
        num_patches = (img_dim[1]//self.patch_size) * (img_dim[2]//self.patch_size)
        if num_patches > self.max_length:
            raise ValueError("The number of patches required to cover the image of size {} is greater than the maximum length of the sequence {}.".format(img_dim, self.max_length))
        # Make sure that the image dimensions are divisible by the patch size:
        # If not, pad the image with zeros:
        # Note: The padding is added equally on all sides of the image.
        # Output an attention mask to ignore the padded tokens:
        if img_dim[1] % self.patch_size != 0:
            pad_h = self.patch_size - img_dim[1] % self.patch_size
        else:
            pad_h = 0
        if img_dim[2] % self.patch_size != 0:
            pad_w = self.patch_size - img_dim[2] % self.patch_size
        else:
            pad_w = 0
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0)

        # Create the attention mask
        attention_mask = torch.ones(x.shape[0], x.shape[2]//self.patch_size, x.shape[3]//self.patch_size)
        if pad_h > 0 or pad_w > 0:
            attention_mask = F.pad(attention_mask, (0, pad_w//self.patch_size, 0, pad_h//self.patch_size), value=0)
        # Flatten the attention mask
        attention_mask = attention_mask.view(x.shape[0], -1)

        # Get the relative x and y position of each patch:
        y_pos = torch.arange(0, x.shape[2], self.patch_size) / (x.shape[2] - 1)
        x_pos = torch.arange(0, x.shape[3], self.patch_size) / (x.shape[3] - 1)
        

        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size) 
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3], x.shape[4], x.shape[5])
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], x.shape[1], -1)        

        if x.shape[1] < self.max_length:
            x = F.pad(x, (0, 0, 0, self.max_length - x.shape[1]), value=0)
            attention_mask = F.pad(attention_mask, (0, self.max_length - attention_mask.shape[1]), value=0)

        #pad y_pos and x_pos with -1 so that they have the same dimension as the attention mask:
        y_pos = F.pad(y_pos, (0, self.max_length - y_pos.shape[0]), value=-1)
        x_pos = F.pad(x_pos, (0, self.max_length - x_pos.shape[0]), value=-1)

        return x, attention_mask, x_pos.unsqueeze(0), y_pos.unsqueeze(0)

    
    def forward(self,x):
        x, attention_mask,x_pos,y_pos = self.to_patch(x)
        x = self.linear(x)        
        return x, attention_mask, x_pos, y_pos
    

class PositionalEmbedding(nn.Module):
    """Adds positional embedding to the input tensor.
    """

    def __init__(self, emb_dim=32, max_length=256 ):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_length = max_length
        self.pos_emb = nn.Parameter(torch.rand(1, max_length, emb_dim))
    def forward(self, x,x_pos, y_pos):
        x = x + self.pos_emb*x_pos.unsqueeze(-1).expand(-1, -1, self.emb_dim) + self.pos_emb*y_pos.unsqueeze(-1).expand(-1, -1, self.emb_dim)
        return x
    

class ResoVit(nn.Module):
# Creates a vision transformer using the PatchEmbedding and PositionalEmbedding classes.

    def __init__(self,
                patch_size=10,
                num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                embedding_dim:int=32, # Hidden size D from Table 1 for ViT-Base
                mlp_size:int=512, # MLP size from Table 1 for ViT-Base
                num_heads:int=4, # Heads from Table 1 for ViT-Base
                attn_dropout:float=0, # Dropout for attention projection
                mlp_dropout:float=0.1, # Dropout for dense/MLP layers 
                embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                max_length=256,
                num_classes=10,
                img_channels=1): 
        super().__init__() # don't forget the super().__init__()!
        self.embedding_dim = embedding_dim
        self.patch_embedding = PatchEmbedding(patch_size=patch_size,max_length=max_length, emb_dim=self.embedding_dim,img_channels=img_channels)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.positional_embedding = PositionalEmbedding(emb_dim=self.embedding_dim,max_length=max_length)
        self.transformer_encoder = nn.TransformerEncoder( 
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=num_heads, dim_feedforward=mlp_size, dropout=attn_dropout),
            num_layers=num_transformer_layers)

            # 10. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            nn.Linear(in_features=self.embedding_dim, 
                    out_features=num_classes)
                                    )
        
    def forward(self, x):
        # x is a list of tensors, I need to get the individual patch embeddings, attn_masks, and positional embeddings and concatenate them into a single tensor before passing it to the transformer encoder.
        for i, img in enumerate(x):
            #Add a batch dimension to img:
            img = img.unsqueeze(0)
            patches, attn_mask, x_pos, y_pos = self.patch_embedding(img)
            # Make sure the attn_mask is on the same device as the patches
            attn_mask = attn_mask.to(patches.device)
            patches = self.embedding_dropout(patches)
            patches = self.positional_embedding(patches,x_pos,y_pos)
            if i == 0:
                x = patches
                attn_masks = attn_mask
            else:
                x = torch.cat((x, patches), dim=0)
                attn_masks = torch.cat((attn_masks, attn_mask), dim=0)
        x = self.transformer_encoder(x, src_key_padding_mask=attn_masks.permute(1,0))
        x = x.mean(dim=1) # Take the average of the embeddings across the sequence dimension
        x = self.classifier(x)
        return x
    




