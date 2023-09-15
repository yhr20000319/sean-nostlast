import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import argparse
import os
import numpy as np
import time
import datetime
import sys
import cv2
import time

#from network.generator import SPADEGenerator
from network.generator_nostlast import SPADEGenerator
from network.models import *
from network.encoder import ConvEncoder
from datasets import *
from network.loss import *
from pretrained.deeplab_jittor.deeplab import DeepLab
# from tensorboardX import SummaryWriter
from Diffaug import * 

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
#注意切换需要加在的模型
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--input_path", type=str, default="datasets/val_B_labels_resized")
parser.add_argument("--img_path", type=str, default="datasets/val_B_labels_resized")
parser.add_argument("--output_path", type=str, default="./results")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
#注意设置状态
parser.add_argument('--status', type=str, default='train')
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--label_c", type=int, default=30, help="number of image channels")
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--nef", type=int, default=32)
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

def save_image(img, path, nrow):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)
    return img



# Loss functions
Gan_loss_g = Ganloss('g')
Gan_loss_d = Ganloss('d')
criterion_pixelwise = nn.L1Loss()
KLD_loss = KLDLoss()
VGG_loss = VGGLoss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 5
lambda_kld = 0.05
lambda_vgg = 10.0
lambda_feat = 10.0
lambda_seg = 40.0

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = SPADEGenerator(opt)
discriminator = Discriminator(opt)
encoder = ConvEncoder(opt)
'''
deeplab = DeepLab(output_stride=16, num_classes=29)
deeplab.eval()
'''


generator.load("./checkpoint/generator_111_367500.pkl")
discriminator.load("./checkpoint/discriminator_111_367500.pkl")
encoder.load("./checkpoint/encoder_111_367500.pkl")
'''
deeplab.load("./pretrained/Epoch_40.pkl")
'''

# Optimizers
optimizer_G = jt.optim.Adam([
                {'params': generator.parameters()},
                {'params': encoder.parameters()}
            ], lr=opt.lr / 2, betas=(opt.b1, opt.b2))
optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=opt.lr * 2, betas=(opt.b1, opt.b2))

# Configure dataloaders
transform_label = [
    transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.NEAREST),
    transform.ToTensor(),
    # transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]

transform_img = [
    transform.Resize(size=(opt.img_height, opt.img_width), mode=Image.BICUBIC),
    transform.ToTensor(),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]



val_dataloader = TestImageDataset(opt.input_path,opt.img_path, transform_label=transform_label, transform_img=transform_img).set_attrs(
    batch_size=1,
    shuffle=False,
    num_workers=1,
)


def get_inst(t):
    edge = jt.init.zero([t.shape[0], 1, t.shape[2], t.shape[3]], int)
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float32()

def change_label(real_A):
    real_A = real_A[:,0,:,:].unsqueeze(1)
    inst_A = get_inst(real_A)
    nc = 29
    input_label = jt.init.zero([real_A.shape[0], nc, real_A.shape[2], real_A.shape[3]])
    real_A = jt.round(real_A).int8()
    temp = jt.init.one([real_A.shape[0], nc, real_A.shape[2], real_A.shape[3]])
    input_label = input_label.scatter_(1, real_A, temp)
    input_label = jt.concat([input_label, inst_A], 1)
    return input_label, real_A[:,0,:,:]

@jt.single_process_scope()
def eval(epoch):
    cnt = 1
    os.makedirs(f"{opt.output_path}", exist_ok=True)
    fake_B_1 = ""
    for i, (real_B,real_B_lable,real_A, photo_id) in enumerate(val_dataloader):
        temp, _ = change_label(real_A)
        real_B_lable , _=change_label(real_B_lable)
        
        st = encoder(real_B, real_B_lable)
        #print(st.shape)
        fake_B = generator(temp, st)
        #这一句必须要放下面
        
        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])
            cnt += 1

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

def divide_pred(pred):
    fake = []
    real = []
    for p in pred:
        fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
        real.append([tensor[tensor.size(0) // 2:] for tensor in p])
    return fake, real

def cal_discriminator(real_A, fake_B, real_B, mode):
    if mode == 'discriminator':
        fake_concat = jt.concat([real_A, fake_B], dim=1).detach()
    elif mode == 'generator':
        fake_concat = jt.concat([real_A, fake_B], dim=1)
    real_concat = jt.concat([real_A, real_B], dim=1)
    fake_and_real = jt.concat([fake_concat, real_concat], dim=0)
    discriminator_out = discriminator(fake_and_real)
    return divide_pred(discriminator_out)
# ----------
#  Testing
# ----------

prev_time = time.time()
eval(201)
