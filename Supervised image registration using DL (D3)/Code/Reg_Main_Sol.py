import SimpleITK as sitk
import torch
from timeit import default_timer

t1 = default_timer()
import sys
sys.path.append("YOUR_CODE_DIRECTORY") 
from uEpdiff import *

from Reg_Tools import *


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)



imagesize = 128 
truncate = 16

net = getUnetT3D()
loadParameter(net, "YOUR_MODEL_WEIGHTS_DIRECTORY/PretrainedReg.tar", device=device)   #e.g., '/home/wn/qs/modelUNET3D/PretrainedReg.tar',
net = net.to(device)

with torch.no_grad():
    #  Load source and target image, and then predict momentum through neural network
    src = sitk.GetArrayFromImage(sitk.ReadImage("YOUR_DATA_DIRECTORY/src3d/src1.mhd")) #(128, 128, 128)
    tar = sitk.GetArrayFromImage(sitk.ReadImage("YOUR_DATA_DIRECTORY/tar3d/tar1.mhd")) #(128, 128, 128)
    src = torch.from_numpy(src).to(device);tar = torch.from_numpy(tar).to(device)

    """ 
    Transpose the axis
    done because I transpose them(source,target,momentum ground truth) during the training process,
    but its not necessary if you don't transpose them during the training process 
    """
    src_trans = src.permute(2,1,0).unsqueeze(0); tar_trans = tar.permute(2,1,0).unsqueeze(0)
    inputs_val = torch.stack((src_trans,tar_trans),dim=1)  ##[1, 2, 128, 128, 128]
    print(inputs_val.shape)
    outputs_val = net(inputs_val)  #[1, 3, 128, 128, 128]
    momentum = outputs_val.permute(0,4,3,2,1)
    # outputs_val = outputs_val.detach().cpu().numpy()  #
    # outputs_val = np.transpose(outputs_val,[0,4,3,2,1])   #1,128,128,128,3

    

##################   start shooting and deform  #######################
## identity transformation
identity = get_grid2(imagesize, device).permute([0,4,3,2,1])  #[1, 3, 128,128,128] ->[1, 128, 128, 128, 3]
Epdiff = Epdiff(device, (truncate, truncate, truncate), (imagesize,imagesize,imagesize), 2.0,1.0,4)  #alpha=2.0, gamma=1.0, lpow=4  


####  momentum to v  #######
velocity = Epdiff.fourier2spatial(Epdiff.Kcoeff * Epdiff.spatial2fourier(momentum[0]))  #[128, 128, 128, 3]
print(velocity.shape)  #[128, 128, 128, 3]
sitk.WriteImage(sitk.GetImageFromArray(velocity.cpu().numpy()), "YOUR_DATA_DIRECTORY/Velocity0.nii.gz")


""" 
    Run geodesic shooting with initial velocity, 
    Results: 10-steps of velocity fields and displacement 
"""
num_steps = 10
v_seq, displacement = Epdiff.forward_shooting_v_and_phiinv(velocity,num_steps)  #displacement: [128, 128, 128, 3]

""" phiinv should added the identity transformation  """
phiinv = displacement.unsqueeze(0) + identity

""" interpolate src with phiinv """
dfm = Torchinterp(src.unsqueeze(0).unsqueeze(0),phiinv) #dfm: [20, 1, 128, 128]

""" save """
sitk.WriteImage(sitk.GetImageFromArray(dfm.cpu().numpy()), "YOUR_DATA_DIRECTORY/DeformedI.nii.gz")


t2 = default_timer()
print("Time: ", t2 - t1)






