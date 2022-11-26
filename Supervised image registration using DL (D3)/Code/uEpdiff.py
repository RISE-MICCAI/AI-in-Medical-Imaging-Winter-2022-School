import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



class Epdiff(nn.Module):
    def __init__(self, device, trunc=(16, 16, 16), iamgeSize=(64,64,64), alpha=3, gamma=1, lpow=6):
        super(Epdiff, self).__init__()
        self.alpha=alpha;self.gamma=gamma; self.lpow=lpow
        self.imgX, self.imgY, self.imgZ, self.iamgeSize = iamgeSize[0],iamgeSize[1],iamgeSize[2],iamgeSize
        self.device = device
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.truncX = trunc[0]
        self.truncY = trunc[1]
        self.truncZ = trunc[2]
        if (self.truncX%2==0):
            self.truncX = self.truncX - 1   #15
        if (self.truncY%2==0):
            self.truncY = self.truncY - 1   #15
        if (self.truncZ%2==0):
            self.truncZ = self.truncZ - 1   #15
        #######   Lcoeff : V˜ → M˜ ∗ maps v to M (momentum)     Kcoeff: M˜ →  V˜
        self.Kcoeff, self.Lcoeff, self.CDcoeff = self.fftOpers (self.alpha, self.gamma, self.lpow, self.truncX, self.truncY, self.truncZ, device)
        # self.Kcoeff2, self.Lcoeff2 = self.fftOpers2 (8)
        
    



    def fftOpers2(self, mode, batchsize):  #shape : [20, 64, 64, 64, 3]
        size_x, size_y, size_z = self.iamgeSize[0], self.iamgeSize[1], self.iamgeSize[2]

        spx = 1 ##spacing information x 
        spy = 1 ##spacing information y 
        spz = 1 ##spacing information z 

        if(self.truncZ != 1):
            gridx = torch.tensor(np.linspace(0, 1-1/size_x, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
            gridy = torch.tensor(np.linspace(0, 1-1/size_y, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
            gridz = torch.tensor(np.linspace(0, 1-1/size_z, size_z), dtype=torch.float)
            gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
            grid = torch.cat((gridx, gridy, gridz), dim=-1).to(self.device)


            trun1 = grid[:, :mode, :mode, :mode, :]     #[b, modes, modes, modes, 3]
            trun2 = grid[:, -mode:, :mode, :mode, :]    #[b, modes, modes, modes, 3]
            trun3 = grid[:, :mode, -mode:, :mode, :]    #[b, modes, modes, modes, 3]
            trun4 = grid[:, -mode:, -mode:, :mode, :]   #[b, modes, modes, modes, 3]
            yy1 = torch.cat((trun1,trun2),dim=-4)       #[b, 2*modes, modes, modes, 3]      #[b, 16, 8, 8, 3]
            yy2 = torch.cat((trun3,trun4),dim=-4)       #[b, 2*modes, modes, modes, 3]      #[b, 16, 8, 8, 3]
            trunr = torch.cat((yy1,yy2),dim=-3)         #[b, 2*modes, 2*modes, modes, 3]    #[b, 16, 16, 8, 3]


            coeff = (-2.0*torch.cos(2.0 * torch.pi * trunr) + 2.0)/(spx*spx);
            val = pow(self.alpha*(torch.sum(coeff,dim=-1))+self.gamma, self.lpow);
            Lcoeff = torch.stack((val,val,val),dim=-1)
            Kcoeff = torch.stack((1/val,1/val,1/val),dim=-1)
        else:
            gridx = torch.tensor(np.linspace(0, 1-1/size_x, size_x), dtype=torch.float)
            gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
            gridy = torch.tensor(np.linspace(0, 1-1/size_y, size_y), dtype=torch.float)
            gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
            grid = torch.cat((gridx, gridy), dim=-1).to(self.device)
            trun1 = grid[:, :mode, :mode, :]     #[b, modes, modes, 3]
            trun2 = grid[:, -mode:, :mode, :]    #[b, modes, modes, 3]
            trunr = torch.cat((trun1,trun2),dim=-3)       #[b, 2*modes, modes, 3]      #[b, 16, 8, 2]
 

            coeff = (-2.0*torch.cos(2.0 * torch.pi * trunr) + 2.0)/(spx*spx)
            val = pow(self.alpha*(torch.sum(coeff,dim=-1))+self.gamma, self.lpow)
            Lcoeff = torch.stack((val,val),dim=-1)       #[20, 16, 8, 2]
            Kcoeff = torch.stack((1/val,1/val),dim=-1)   #[20, 16, 8, 2]

          

        return Kcoeff, Lcoeff



    def fftOpers2D(self, mode, batchsize):  #shape : [20, 64, 64, 64, 3]
        size_x, size_y, size_z = self.iamgeSize[0], self.iamgeSize[1], self.iamgeSize[2]

        gridx = torch.tensor(np.linspace(0, 1-1/size_x, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])

        gridy = torch.tensor(np.linspace(0, 1-1/size_y, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

        # gridz = torch.tensor(np.linspace(0, 1-1/size_z, size_z), dtype=torch.float)
        # gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        # grid = torch.cat((gridx, gridy, gridz), dim=-1).to(self.device)
        grid = torch.cat((gridx, gridy), dim=-1).to(self.device)


        trun1 = grid[:, :mode, :mode, :]     #[b, modes, modes, 3]
        trun2 = grid[:, -mode:, :mode, :]    #[b, modes, modes, 3]
        
        trunr = torch.cat((trun1,trun2),dim=-3)       #[b, 2*modes, modes, 3]      #[b, 16, 8, 8, 3]


        spx = 1 ##spacing information x 
        spy = 1 ##spacing information y 
        spz = 1 ##spacing information z 

        coeff = (-2.0*torch.cos(2.0 * torch.pi * trunr) + 2.0)/(spx*spx)
        val = pow(self.alpha*(torch.sum(coeff,dim=-1))+self.gamma, self.lpow)
        Lcoeff2D = torch.stack((val,val),dim=-1)
        Kcoeff2D = torch.stack((1/val,1/val),dim=-1)

        return Lcoeff2D, Kcoeff2D







    def fftOpers (self, alpha, gamma, lpow, truncX, truncY, truncZ, device):
        fsx = self.iamgeSize[0]
        fsy = self.iamgeSize[1]
        fsz = self.iamgeSize[2]

        size = truncX*truncY*truncZ 
        fsize = fsx * fsy * fsz 
        beginX = int(-(truncX-1) / 2.0);  #-7
        beginY = int(-(truncY-1) / 2.0);  #-7
        beginZ = int(-(truncZ-1) / 2.0);  #-7
        endX = beginX + truncX;           #8
        endY = beginY + truncY;           #8
        endZ = beginZ + truncZ;           #8
        padX = 2 * truncX - 1; 
        padY = 2 * truncY - 1;
        padZ = 2 * truncZ - 1;
        padsize = padX * padY * padZ;

        sX = 2.0 * torch.pi / fsx;
        sY = 2.0 * torch.pi / fsy;
        sZ = 2.0 * torch.pi / fsz;
        id = 0;
        fftLoc = torch.zeros(3 * size).to(device);
        corrLoc = torch.zeros(3 * size).to(device);
        conjLoc = torch.zeros(3 * size).to(device);
        ################################ Kenerls ###################################
        for k in range (beginZ, endZ):
            for p in range (beginY, endY):
                for i in range (beginX, endX):
                    fftZ = k; corrZ = k; conjZ = k;
                    fftY = p; corrY = p; conjY = p; 
                    fftX = i; corrX = i; conjX = i;

                    if(k < 0):
                        fftZ = k + fsz; corrZ = k + padZ; conjZ = fftZ;
                    if(p < 0):
                        fftY = p + fsy; corrY = p + padY; conjY = fftY;
                    if(i < 0): 
                        fftX = i + fsx; corrX = i + padX;
                        conjX = -i ; conjY = -p; conjZ = -k;
                        if (p > 0):
                            conjY = -p + fsy;
                        if (k > 0):
                            conjZ = -k + fsz;
                    fftLoc[3*id] = fftX; corrLoc[3*id] = corrX; conjLoc[3*id] = conjX;
                    fftLoc[3*id+1] = fftY; corrLoc[3*id+1] = corrY; conjLoc[3*id+1] = conjY;
                    fftLoc[3*id+2] = fftZ; corrLoc[3*id+2] = corrZ; conjLoc[3*id+2] = conjZ;
                    id +=1
        Lcoeff = torch.zeros(truncX*truncY*truncZ*3, dtype=torch.cfloat).to(device)
        Kcoeff = torch.zeros(truncX*truncY*truncZ*3, dtype=torch.cfloat).to(device)
        CDcoeff = torch.zeros(truncX*truncY*truncZ*3, dtype=torch.cfloat).to(device)
        spx = 1 ##spacing information x 
        spy = 1 ##spacing information y 
        spz = 1 ##spacing information z 
    




        for id in range (0,size):
            index = 3*id;
            xcoeff = (-2.0*torch.cos(sX*fftLoc[index]) + 2.0)/(spx*spx);
            ycoeff = (-2.0*torch.cos(sY*fftLoc[index+1]) + 2.0)/(spy*spy);
            zcoeff = (-2.0*torch.cos(sZ*fftLoc[index+2]) + 2.0)/(spz*spz);
            val = pow(alpha*(xcoeff + ycoeff + zcoeff)+gamma, lpow);
            
            Lcoeff[index]= val
            Lcoeff[index+1] = val
            Lcoeff[index+2] = val
            Kcoeff[index] = 1/val
            Kcoeff[index+1] = 1/val
            Kcoeff[index+2] = 1/val
            CDcoeff[index]= complex(0,torch.sin(sX*fftLoc[index])/spx)
            CDcoeff[index+1]= complex(0,torch.sin(sX*fftLoc[index+1])/spy)
            CDcoeff[index+2]= complex(0,torch.sin(sX*fftLoc[index+2])/spz)
        
        
        return Kcoeff, Lcoeff, CDcoeff




    '''low-dimensional fftn/ifftn transforms for computation'''
    def fourier2spatial_bandlimi(self, input_vec):
        scratch = input_vec.reshape(self.truncX, self.truncY, self.truncZ, 3)
        scratch_sp = torch.zeros(self.truncX, self.truncY, self.truncZ, 3,dtype=torch.cfloat)
        for i in range (0,3):
            # scratch_sp[...,i] = torch.fft.ifftn(torch.fft.ifftshift(scratch[:,:,i].reshape(truncX, truncY, truncZ)), dim=(-3,-2,-1))*truncX*truncY*truncZ
            scratch_sp[...,i] = torch.fft.ifftn(torch.fft.ifftshift(scratch[...,i].reshape(self.truncX, self.truncY, self.truncZ), dim=(-3,-2,-1)), dim=(-3,-2,-1))*(self.truncX*self.truncY*self.truncZ)
            # scratch_sp[...,i] = torch.fft.ifftn(scratch[...,i].reshape(truncX, truncY, truncZ))

        return scratch_sp
    def spatial2fourier_bandlimi(self,input_vec):
        [m,n,q,c]=input_vec.shape
        scracth_cp = torch.zeros(self.truncX,self.truncY,self.truncZ, 3 ,dtype=torch.cfloat).to(self.device)
        for i in range (0,c):
            scracth_cp[...,i]=torch.fft.fftshift(torch.fft.fftn((input_vec[...,i].reshape(m,n,q)), dim=(-3,-2,-1)), dim=(-3,-2,-1))/(self.truncX*self.truncY*self.truncZ)
            # scracth_cp[...,i]=torch.fft.fftn((input_vec[...,i].reshape(m,n,q)))
        return scracth_cp


    ################################ Computational operations ###################################
    '''Jacobian and JacobianTranspose'''
    def Jacobian (self, CDcoeff_oper, input_vec):
        size = self.truncX*self.truncY*self.truncZ;
        JacX = torch.zeros(self.truncX*self.truncY*self.truncZ*3 ,dtype=torch.cfloat).to(self.device)
        JacY = torch.zeros(self.truncX*self.truncY*self.truncZ*3 ,dtype=torch.cfloat).to(self.device)
        JacZ = torch.zeros(self.truncX*self.truncY*self.truncZ*3 ,dtype=torch.cfloat).to(self.device)
        for i in range (0, size): 
            index = 3*i;
            JacX[index] = CDcoeff_oper[index]*input_vec[index];
            JacX[index+1] = CDcoeff_oper[index]*input_vec[index+1];
            JacX[index+2] = CDcoeff_oper[index]*input_vec[index+2];

            JacY[index] = CDcoeff_oper[index+1]*input_vec[index];
            JacY[index+1] = CDcoeff_oper[index+1]*input_vec[index+1];
            JacY[index+2] = CDcoeff_oper[index+1]*input_vec[index+2];

            JacZ[index] = CDcoeff_oper[index+2]*input_vec[index];
            JacZ[index+1] = CDcoeff_oper[index+2]*input_vec[index+1];
            JacZ[index+2] = CDcoeff_oper[index+2]*input_vec[index+2];
        return JacX, JacY, JacZ
    def JacobianTranspose (self,CDcoeff_oper, input_vec):
        size = self.truncX*self.truncY*self.truncZ;
        JacX = torch.zeros(self.truncX*self.truncY*self.truncZ*3 ,dtype=torch.cfloat).to(self.device)
        JacY = torch.zeros(self.truncX*self.truncY*self.truncZ*3 ,dtype=torch.cfloat).to(self.device)
        JacZ = torch.zeros(self.truncX*self.truncY*self.truncZ*3 ,dtype=torch.cfloat).to(self.device)
        for i in range (0, size): 
            index = 3*i;
            JacX[index] = CDcoeff_oper[index]*input_vec[index];
            JacX[index+1] = CDcoeff_oper[index+1]*input_vec[index];
            JacX[index+2] = CDcoeff_oper[index+2]*input_vec[index];

            JacY[index] = CDcoeff_oper[index]*input_vec[index+1];
            JacY[index+1] = CDcoeff_oper[index+1]*input_vec[index+1];
            JacY[index+2] = CDcoeff_oper[index+2]*input_vec[index+1];

            JacZ[index] = CDcoeff_oper[index]*input_vec[index+2];
            JacZ[index+1] = CDcoeff_oper[index+1]*input_vec[index+2];
            JacZ[index+2] = CDcoeff_oper[index+2]*input_vec[index+2];
        return JacX, JacY, JacZ


    
    def fourier2spatial(self, input_vec):
    
        '''input: a low-dimensional velocity as an 1-d vector'''
        input_vec = input_vec.reshape(self.truncX, self.truncY, self.truncZ, 3)
        [m,n,q,c]=input_vec.shape
        start_x = int(self.imgX/2 - m/2)+1; end_x = int(self.imgX/2 + m/2)+1
        start_y = int(self.imgY/2 - n/2)+1; end_y =int(self.imgY/2 + n/2)+1
        start_z = int(self.imgZ/2 - q/2)+1; end_z = int(self.imgZ/2 + q/2)+1
        
        
        output_vec =  torch.zeros(self.imgX, self.imgY, self.imgZ, 3).to(self.device)
        for i in range (0, c):
            large_pad = torch.zeros(self.imgX, self.imgY,self.imgZ,dtype=torch.cfloat).to(self.device)
            large_pad[start_x:end_x, start_y:end_y, start_z:end_z] =input_vec[...,i].reshape(m,n,q)
            tt = torch.fft.ifftn(torch.fft.fftshift(large_pad, dim=(-3,-2,-1)), dim=(-3,-2,-1))*self.imgX*self.imgY*self.imgZ
            # print(torch.mean(tt.real), "   ", torch.mean(tt.imag))
            output_vec[...,i] = tt
        '''output: a high-dimensional [h,w,c] tensor'''
        return output_vec



    def spatial2fourier (self, input_vec):
        '''input vect as a high-dimensional [h,w,c] tensor'''
        [m,n,q,c ]=input_vec.shape
        start_ = int(m/2 - self.truncX/2)+1
        end_  = int(n/2 + self.truncX/2)+1

        """ x_di =input_vec[...,0].reshape(m,n,q)
        y_di =input_vec[...,1].reshape(m,n,q)
        z_di =input_vec[...,2].reshape(m,n,q)
        velo_f_x = torch.fft.fftshift(torch.fft.fftn(x_di, dim=(-3,-2,-1)), dim=(-3,-2,-1))/(m*n*q)
        velo_f_y = torch.fft.fftshift(torch.fft.fftn(y_di, dim=(-3,-2,-1)), dim=(-3,-2,-1))/(m*n*q)
        velo_f_z = torch.fft.fftshift(torch.fft.fftn(z_di, dim=(-3,-2,-1)), dim=(-3,-2,-1))/(m*n*q)

        low_dim_velo = torch.zeros(self.truncX*self.truncY*self.truncZ*3 ,dtype=torch.cfloat).to(self.device)

        low_velo_f_x = velo_f_x[start_:end_,start_:end_,start_:end_].reshape(self.truncX*self.truncY*self.truncZ);
        low_velo_f_y = velo_f_y[start_:end_,start_:end_,start_:end_].reshape(self.truncX*self.truncY*self.truncZ,);
        low_velo_f_z = velo_f_z[start_:end_,start_:end_,start_:end_].reshape(self.truncX*self.truncY*self.truncZ);
        size = self.truncX*self.truncY*self.truncZ;
        
        for i in range (0, size): 
            index = 3*i;
            low_dim_velo[index] = low_velo_f_x [i]
            low_dim_velo[index+1] = low_velo_f_y [i]
            low_dim_velo[index+2] = low_velo_f_z [i]
        '''Low-dimensional velocity as an 1-d vector''' """


        velo_f = torch.fft.fftshift(torch.fft.fftn(input_vec, dim=(-4,-3,-2)), dim=(-4,-3,-2))/(m*n*q)
        low_velo_f = velo_f[start_:end_,start_:end_,start_:end_,...]
        low_velo_f = torch.flatten(low_velo_f,0,-2)
        low_velo_f = low_velo_f.flatten()



        return low_velo_f



    def mul_vec_corr(self, input_vect1, input_vect2,flag):
        '''Perform correlation if flag equals to 1'''
        if (flag == 1):
            input_vect1.imag *=-1
        if (flag == 0):
            input_vect1.imag *=1
        return torch.mul(input_vect1, input_vect2)

    ##(Dv)^T)*m and (Dv)*\phi
    def complex_corr(self, in_velo, in_momen, CDcoeff, flag): #in_velo:u     in_momen:velocity (not momentum)
        if (flag == 1):
            jx,jy,jz = self.JacobianTranspose(CDcoeff,in_velo)
        if (flag == 0):
            jx,jy,jz = self.Jacobian(CDcoeff,in_velo)

        momen_sp = self.fourier2spatial_bandlimi(in_momen)
        jx_sp = self.fourier2spatial_bandlimi(jx)    
        jy_sp = self.fourier2spatial_bandlimi(jy)
        jz_sp = self.fourier2spatial_bandlimi(jz)
        velo_sp =  self.fourier2spatial_bandlimi(in_velo)


        """ temp1 = mul_vec_corr(jx_sp, velo_sp, flag);
        temp2 = mul_vec_corr(jy_sp, velo_sp, flag);
        temp3 = mul_vec_corr(jz_sp, velo_sp, flag); """

        #######     nellie    ########
        temp1 = self.mul_vec_corr(jx_sp, momen_sp, flag);
        temp2 = self.mul_vec_corr(jy_sp, momen_sp, flag);
        temp3 = self.mul_vec_corr(jz_sp, momen_sp, flag);


        DvTranm_sp = torch.zeros(self.truncX,self.truncY,self.truncZ,3,dtype=torch.cfloat).to(self.device)
        DvTranm_sp[...,0] = torch.sum(temp1,dim=-1)
        DvTranm_sp[...,1] = torch.sum(temp2,dim=-1)
        DvTranm_sp[...,2] = torch.sum(temp3,dim=-1)

        
        #     print (temp1 )
        DvTranm_grid = self.spatial2fourier_bandlimi(DvTranm_sp)
        DvTranm =DvTranm_grid.reshape(self.truncX*self.truncY*self.truncZ*3)
        return DvTranm 


    def adjoint (self, in_velo, flag):
        momen = self.Lcoeff*in_velo                        
        DvTranm = self.complex_corr(in_velo, momen, self.CDcoeff,1)
        div_mconvv =  self.complex_conv(in_velo, momen)
        # dv = -self.Kcoeff*(DvTranm)
        dv = self.Kcoeff*(DvTranm+ div_mconvv) 
        # dv = self.Kcoeff*(DvTranm)
        return dv

    def div_CD (self, input_vect):
        jx,jy,jz = self.Jacobian(self.CDcoeff,input_vect)
        div_mv=jx+jy+jz
        return div_mv

    def multi_tensor(self, inputvec):
        outputvec = inputvec.view(1,1,3,1).repeat(1,1,1,1)
        return outputvec 
    def complex_conv_basic (self, input_vec1, input_vec2):
        #input_vec1 and input_vec2 should be complex-valued 3X1 torch.tensor vectors
        real_1 = self.multi_tensor ((input_vec1.real))
        imag_1 = self.multi_tensor (input_vec1.imag)
        real_2 = self.multi_tensor (input_vec2.real)
        imag_2 = self.multi_tensor (input_vec2.imag)
        #compute convolution for real parts (checked)
        real_1=torch.flip(real_1,dims=[1,2])
        imag_2=torch.flip(imag_2,dims=[1,2])
        conv_val_rr = F.conv1d(real_2,real_1,padding = 1)
        conv_val_ii = F.conv1d(imag_1,imag_2,padding = 1)
        
        #compute convolution for imag parts (checked)
        real_1=torch.flip(real_1,dims=[1,2])
        real_2=torch.flip(real_2,dims=[1,2])
        conv_val_ri = F.conv1d(real_1,imag_2,padding = 1)
        conv_val_ir= F.conv1d(imag_1,real_2,padding = 1)
        
        conv_val_r = conv_val_rr - conv_val_ii
        conv_val_i = conv_val_ri + conv_val_ir

        real_vec = conv_val_r[:,:,:,1]
        imag_vec = conv_val_i[:,:,:,1]
        
        re= torch.tensor([complex(real_vec[:,:,0],imag_vec[:,:,0]),complex(real_vec[:,:,1],imag_vec[:,:,1]),complex(real_vec[:,:,2],imag_vec[:,:,2])])
        return re
    def complex_conv(self,in_velo, in_momen):
        momen_sp = self.fourier2spatial_bandlimi(in_momen)
        velo_sp =  self.fourier2spatial_bandlimi(in_velo)
        mv_sp = torch.zeros(self.truncX,self.truncY,self.truncZ,3,dtype=torch.cfloat)
        for i in range (0, self.truncX):
            for j in range (0, self.truncY):
                for k in range (0, self.truncZ):
                    vec1= momen_sp[i,j,k,:]     #[3]
                    vec2= velo_sp[i,j,k,:]      #[3]
                    mv_sp[i,j,k,:] = self.complex_conv_basic(vec1,vec2)
        #mv_sp = complex_conv_basic(momen_sp, velo_sp);
        mv_grid = self.spatial2fourier_bandlimi(mv_sp)
        mv = mv_grid.reshape(self.truncX*self.truncY*self.truncZ*3)
        div_val = self.div_CD(mv)
        return div_val


    def forward_shooting(self, u0, v0spatial, dt=-0.1):
        v0spatial = v0spatial.permute(0,2,3,4,1)[0]
        v0 = self.spatial2fourier(v0spatial) #[10125]  3*15*15*15
        #######   Lcoeff : V˜ → M˜ ∗ maps v to M (momentum)     Kcoeff: M˜ →  V˜
        # v0 = Kcoeff*v0                                  #[10125]  3*15*15*15
        du = self.complex_corr(u0,v0, self.CDcoeff,0)
        u0 = u0 + (du+v0)*dt
        return u0


        

    def forward_shooting_v(self,in_velo,num_steps):
        if in_velo.shape[0]==1:
            v0spatial = in_velo.permute(0,2,3,4,1)[0]  # move channel to the last dimension
        else:
            v0spatial = in_velo

        v0 = self.spatial2fourier(v0spatial) #[10125]  3*15*15*15
        dt = 1/num_steps
        v_seq = []
        for Eu_step in range (0, num_steps-1):                   
            dv = self.adjoint(v0,1)
            # v0 = v0+dv*dt
            v0 = v0-dv*dt
            v_seq.append(self.fourier2spatial(v0))
        return v_seq


    def forward_shooting_v_and_phiinv(self,in_velo,num_steps):
        if in_velo.shape[0]==1:
            v0spatial = in_velo.permute(0,2,3,4,1)[0]  # move channel to the last dimension
        else:
            v0spatial = in_velo

        v0 = self.spatial2fourier(v0spatial) #[10125]  3*15*15*15
        dt = 1/num_steps
        v_seq = []
        v_seq_fre = []
        for Eu_step in range (0, num_steps-1):                   
            dv = self.adjoint(v0,1)
            # v0 = v0+dv*dt
            v0 = v0-dv*dt
            v_seq.append(self.fourier2spatial(v0))
            v_seq_fre.append(v0)

        u0 = torch.zeros(v0.shape, dtype=torch.cfloat).to(self.device)
        for Eu_step in range (0, num_steps-1):                   
            du = self.complex_corr(u0,v_seq_fre[Eu_step], self.CDcoeff,0)
            u0 = u0 - (du+v_seq_fre[Eu_step])*dt

        SpatialU0 = self.fourier2spatial(u0)
        return v_seq, SpatialU0
        
