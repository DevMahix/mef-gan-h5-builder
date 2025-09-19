# MEF GAN h5 Dataset Builder

Make sure you create the following folder structure for training and label the images accordingly.  
Each subfolder should contain the images that correspond to its label.

The dirs inside create_h5_file/create_h5_file.m are absolute path, make sure you update it accordingly (Was lazy to figure out to make it relative in MATLAB). The entry point is the above mentioned file.

```
training/
├── gt/ # images for the "gt" class
├── under/ # images for the "under" class
└── over/ # images for the "over" class
```