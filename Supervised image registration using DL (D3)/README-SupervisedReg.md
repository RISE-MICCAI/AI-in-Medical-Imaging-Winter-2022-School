## Requirements

- Pytorch 1.10.1 

  Support both CUDA and CPU, you can install pytorch 1.10.1 with the following command (more information can be found in the link:   https://pytorch.org/get-started/previous-versions/):

  ```
  #Linux and Windows
  
  # CUDA 10.2
  conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
  
  # CUDA 11.3
  conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
  
  # CPU Only
  conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cpuonly -c pytorch
  ```

- SimpleITK 2.2.0

- numpy 22.2.2

    

## Instructions

- Reg_Main.py is the main tool to first predict the transformations between pairwise images, followed by deforming the source image by a geodesic shooting algorithm;
  ***Note the provided pre-trained registration model ('PretrainedReg.tar') is on 3D brain images with the size of $128^3$.

- Reg_Tools.py includes some utility functions;
- uEpdiff.py is the main code of forward shooting by FLASH;

