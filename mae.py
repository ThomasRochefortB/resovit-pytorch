from einops import repeat
import torch.nn.functional as F

class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio = 0.75,
        decoder_depth = 1,
        decoder_heads = 8,
        decoder_dim_head = 64
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio
        self.encoder = encoder
        self.decoder_dim = decoder_dim
        num_patches, encoder_dim = encoder.positional_embedding.pos_emb.shape[-2:]
        pixel_values_per_patch = encoder.patch_embedding.linear.weight.shape[1]

        self.decoder = encoder.transformer_encoder
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))

        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)
        self.patch_to_emb = encoder.patch_embedding.linear

    def forward(self, x):
    # x is a list of tensors, I need to get the individual patch embeddings, attn_masks, and positional embeddings and concatenate them into a single tensor before passing it to the transformer encoder.
        for i, img in enumerate(x):
            #Add a batch dimension to img:
            img = img.unsqueeze(0)
            patches,attn_mask, x_pos, y_pos = self.encoder.patch_embedding.to_patch(img)
        
            if i == 0:
                x = patches
                attn_masks = attn_mask
            else:
                x = torch.cat((x, patches), dim=0)
                attn_masks = torch.cat((attn_masks, attn_mask), dim=0)
                
        batch, num_patches, num_pixels = x.shape
        patch_emb = self.patch_to_emb(x)
        patch_emb_shape = patch_emb.shape
        # assume patch_embedding and attention_mask are already defined
        # flatten the patch embedding and attention mask tensors
        patch_embedding_flat = patch_emb.view(-1, self.encoder.embedding_dim)
        attention_mask_flat = attn_masks.view(-1)

        # create a boolean mask for non-padded patches
        non_pad_mask = ~torch.eq(attention_mask_flat, 0)

        # compute the number of non-padded patches to replace with noise
        n_replace = int(non_pad_mask.sum() * self.masking_ratio)

        # generate a random permutation of non-padded patch indices
        non_pad_indices = torch.nonzero(non_pad_mask).flatten()
        perm = torch.randperm(non_pad_indices.size(0))

        # select the first n_replace indices from the permutation
        replace_indices = non_pad_indices[perm[:n_replace]]

        # replace selected patches with noise
        noise = torch.randn((n_replace, self.encoder.embedding_dim), device=patch_emb.device)
        patch_embedding_flat[replace_indices] = noise

        # reshape the modified patch embedding tensor to the original shape
        patch_embedding_modified = patch_embedding_flat.view(patch_emb_shape)
        #Apply positional embedding:
        tokens = model.positional_embedding(patch_embedding_modified)

        # create a boolean mask for non-padded and non-masked patches
        non_pad_mask = ~torch.eq(attention_mask_flat, 0)
        non_mask_mask = ~torch.eq(attention_mask_flat, -1)

        # create a boolean mask for the replaced patches
        replace_mask = torch.zeros_like(attention_mask_flat)
        replace_mask[replace_indices] = 1
        replace_mask = replace_mask.bool() # convert to Boolean-type tensor

        # create the final mask tensor by combining the non-pad, non-mask, and non-replace masks
        mask = non_pad_mask & non_mask_mask & ~replace_mask

        # reshape the mask tensor to the original shape
        mask = mask.view(patch_emb_shape[0], patch_emb_shape[1])
        replace_mask = replace_mask.view(patch_emb_shape[0],patch_emb_shape[1])

        encoded_tokens = model.transformer_encoder(tokens,src_key_padding_mask=mask.permute(1,0))
        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens
        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(torch.rand(batch, num_patches, device = decoder_tokens.device).argsort(dim = -1))
                
    
        decoded_tokens = self.decoder(unmasked_decoder_tokens)
        pred_pixel_values = self.to_pixels(decoded_tokens)

        predictions = pred_pixel_values*replace_mask.unsqueeze(-1).expand(-1, -1, num_pixels)
        actuals = x*replace_mask.unsqueeze(-1).expand(-1, -1, num_pixels)

        return predictions, actuals, x, attn_mask