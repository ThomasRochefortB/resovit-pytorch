# resovit-pytorch
Implementation of a variable resolution image pipeline for training Vision  Transformers in PyTorch. The model can ingest images with varying resolutions without the need for preprocessing steps such as resizing and padding to a common size.

* The maximum size of the image that the ViT model can ingest is defined by the max_length parameter. Any image with less than max_length patches are padded with empty patches which are masked for the attention layers.

For example, you can train the model on the Oxford 102 Flowers dataset which consists of various flower images of varying (H,W) dimensions
