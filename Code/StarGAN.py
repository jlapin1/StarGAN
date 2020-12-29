"""
"Failed to get convolutional algorithm"" is an INTERMITTENT issue

Had issues until I realized the last layer was still applying InstNorm and ReLU before Tanh

Try intializing weights of kernels to something much smaller
    ~ 1/sqrt(in_channels*kernel_size[0]*kernel_size[1])
"""
RSZ = 128
BATCH_SIZE = 16
LAMBDA_CLS = 1
LAMBDA_REC = 10
N_CRITIC = 1
GAMMA = 0.5
LN = 0.0

from PIL import Image
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import time
import sys
sys.path.append('C:/Users/glapi/Documents/Python/Tensorflow/MyProjects/Lib/')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# https://github.com/tensorflow/tensorflow/issues/24828
#https://github.com/tensorflow/tensorflow/issues/28081
config=ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

##############################
######## Datasets ############
##############################

def normalize(im):
    im = (im-127.5)/127.5
    return im

def random_crop_and_jitter(im, rsz):
    addon = int(0.1171875*rsz)
    im = im.resize((rsz+addon, rsz+addon))
    
    rand = np.random.randint(addon, size=(2,))
    im = im.crop((rand[0], rand[1], rsz+rand[0], rsz+rand[1]))
    
    if np.random.rand()<0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    
    return im

def download(filenames, path):
    out = []
    for m in filenames:
        # Read raw image in
        im = Image.open(path+m)
        # Crop long dimension to short
        (w,h) = im.size
        if w<h:
            crp = (h-w)//2
            im = im.crop((0, crp, w, h-crp))
        else:
            crp = (w-h)//2
            im = im.crop((crp, 0, w-crp, h))
        # Resize
        #im = im.resize((256, 256))
        # Random crop and jitter
        #im = random_crop_and_jitter(im)
        # Normalize
        #im = normalize(im)
        out.append(im)
    # return a list of PILs
    return out       

def dataset(pils, rsz):
    # list(pils) -> np.array -> tf.constant
    out = np.zeros((len(pils), rsz, rsz, 3))
    for m,n in enumerate(pils):
        n = n.resize((rsz, rsz))
        # Random crop and jitter
        # n = random_crop_and_jitter(n, rsz)
        out[m] = normalize(np.array(n))
    out = tf.constant(out, tf.float32)
    return out

def disk_to_tensor(Files, Path, Rsz):
    images = download(Files, Path)
    tensor = dataset(images, Rsz)
    return tensor

def annotations():
    f = open("C:/Users/glapi/MyDatasets/celebA/annotations.txt", 'r')
    length = int(f.readline())
    labels = f.readline().split()
    # cut down to 5 binary attributes: black hair, blond hair, brown hair, male, young
    labels = np.array(labels)[[8, 9, 11, 20, 39]]
    dic = {m:n for m,n in enumerate(labels)}
    ann = []
    for line in f:
        ann.append([(int(m)==1)*1 for m in line.split()[1:]])
    return np.array(ann)[:, [8, 9, 11, 20, 39]], labels, dic

initializer = tf.keras.initializers.GlorotUniform()#tf.random_normal_initializer(0., 0.02)

ann, labels, dic = annotations()
PATH = 'C:/Users/glapi/MyDatasets/celebA/img_align_celeba/'
files = os.listdir(PATH)
SZ = len(files)
ND = len(labels)
files = files[:SZ]

# Fixed test images
fixed_files = ['000101.jpg', '000201.jpg', '000301.jpg', '000401.jpg']
fixed_ann = ann[[100, 200, 300, 400]]
# fixed_targ = 1*(fixed_ann==False)
hold = np.loadtxt('fixed_ann_targ.csv', delimiter=',').astype('float32')
fixed_ann = tf.constant(hold[:4], tf.float32)
fixed_targ = tf.constant(hold[4:], tf.float32)
FIXED_INP = disk_to_tensor(fixed_files, PATH, RSZ)

# Shuffle
perm = np.random.permutation(SZ)
files = np.array(files)[perm]
ann = ann[perm]
Ann = tf.constant(ann, tf.float32)

###############################
########### Models ############
###############################
def c7s1(k, norm=True, act=True):
    model = tf.keras.Sequential(name='c7s1-%d'%(k))
    bias = True if norm else False
    model.add(tf.keras.layers.Conv2D(k, 7, 1, padding='SAME', 
                               kernel_initializer=initializer, use_bias=bias))
    if norm:
        model.add(tfa.layers.InstanceNormalization())
    if act:
        model.add(tf.keras.layers.ReLU())
    return model

def dk(k):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(k, 3, 2, padding='SAME',
                               kernel_initializer=initializer, use_bias=False),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.ReLU()
        ], name='d%d'%(k))

class Rk(tf.keras.layers.Layer):
    def __init__(self, k, name=''):
        super(Rk, self).__init__()
        self._name = 'R%d'%(k) + name
        self.k = k
        self.block = tf.keras.Sequential([
            tf.keras.layers.Conv2D(k, 3, 1, padding='SAME',
                                   kernel_initializer=initializer, use_bias=False),
            tfa.layers.InstanceNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(k, 3, 1, padding='SAME',
                                   kernel_initializer=initializer, use_bias=False),
            tfa.layers.InstanceNormalization()
            ])
    
    def build(self, x):
        if x[-1]!=self.k:
            self.shortcut = tf.keras.layers.Conv2D(self.k, 1, 1, padding='SAME',
                                                   kernel_initializer=initializer, use_bias=True)
        else:
            self.shortcut = tf.keras.activations.linear
    def call(self, x):
        return self.shortcut(x) + self.block(x)

def uk(k):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(k, 3, 2, padding="SAME",
                                        kernel_initializer=initializer, use_bias=False),
        tfa.layers.InstanceNormalization(),
        tf.keras.layers.ReLU()
        ], name='u%d'%(k))

def Generator(nblocks=6):
    image = tf.keras.layers.Input(shape=(RSZ, RSZ, 3))
    mask = tf.keras.layers.Input(shape=(ND,))
    
    inp = tf.multiply(tf.ones((RSZ, RSZ, ND)), 
                      mask[:, tf.newaxis, tf.newaxis, :]) # RSZ, RSZ, ND
    inp = tf.keras.layers.Concatenate()([image, inp]) # RSZ, RSZ, 3+ND
    
    out = c7s1(64)(inp) # RSZ, RSZ, 64
    out = dk(128)(out) # RSZ/2, RSZ/2, 128
    out = dk(256)(out) # RSZ/4, RSZ/4, 256
    for m in range(nblocks):
        out = Rk(256, name='_%d'%(m))(out) # RSZ/4, RSZ/4, 256
    out = uk(128)(out) # RSZ/2, RSZ/2, 128
    out = uk(64)(out) # RSZ, RSZ, 64
    out = c7s1(3, norm=False, act=False)(out)
    out = tf.keras.activations.tanh(out)
    return tf.keras.Model(inputs=[image, mask], outputs=out)

def Ck(k, apply_norm=False, stride=2, pad='SAME'):
    model = tf.keras.Sequential()
    bias=False if apply_norm else True
    model.add(tf.keras.layers.Conv2D(k, 4, stride, padding=pad,
                                     kernel_initializer=initializer, use_bias=bias))
    if apply_norm:
        model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.01))
    return model

def Discriminator(nd):
    inp = tf.keras.layers.Input(shape=(RSZ, RSZ, 3))
    out = Ck(64)(inp)
    out = Ck(128)(out)
    out = Ck(256)(out)
    out = Ck(512)(out)
    out = Ck(1024)(out)
    out = Ck(2048)(out)
    
    out1 = tf.keras.layers.Conv2D(1, 3, 1, padding='SAME', activation='linear',
                                  kernel_initializer=initializer, use_bias=False)(out)
    out2 = tf.keras.layers.Conv2D(nd, 2, 1, padding='VALID', activation='linear',
                                  kernel_initializer=initializer, use_bias=False)(out)
    
    out2 = tf.squeeze(out2)
    return tf.keras.Model(inputs=inp, outputs=[out1, out2])

G = Generator()
D = Discriminator(ND)

########################################
########## Loss function ###############
########################################
# @tf.function
def recon_loss(orig, gened):
    return tf.reduce_mean(tf.abs(orig-gened))

def gradpen(inp):
    with tf.GradientTape() as tape:
        tape.watch(inp)
        [dreal, realcls] = D(inp, training=True)
    Rgrads = tape.gradient(dreal, inp)
    Rgrads = tf.reshape(Rgrads, (Rgrads.shape[0], -1,))
    R1 = tf.reduce_mean(tf.norm(Rgrads, axis=1))
    return dreal, realcls, R1

def model_loss(inp, fake, real_label, fake_label):    
    """
    Binary crossentropy treats every column of a single row as independent 
    until an average is taken across the row at the end
    
    average( lab_1*ln(logit_1), lab_2*ln(logit_2), ... )
    
    Categorical crossentropy sums across the row
    
    sum( lab_1*ln(logit_1), lab_2*ln(logit_2), ... )
    """
    
    
    Dreal,realcls,R1 = gradpen(inp)
    [Dfake,fakecls] = D(fake)
    # 1. Adversarial loss
    
    glabel = tf.ones_like(Dfake)#tf.random.uniform((Dfake.shape), 1-LN, 1)
    dlabelr = tf.ones_like(Dreal)#tf.random.uniform((Dreal.shape), 1-LN, 1)
    dlabelf = tf.zeros_like(Dfake)#tf.random.uniform((Dfake.shape), 0, LN)
    
    
    
    # D has no sigmoid activation: "from_logits=True"
    real_loss = tf.keras.losses.binary_crossentropy(
        dlabelr, Dreal, from_logits=True)
    real_loss = tf.reduce_mean(real_loss)
    
    fake_loss = tf.keras.losses.binary_crossentropy(
        dlabelf, Dfake, from_logits=True)
    fake_loss = tf.reduce_mean(fake_loss)
    
    Dadv = 0.5*(real_loss+fake_loss)
    
    Gadv = tf.keras.losses.binary_crossentropy(
        glabel, Dfake, from_logits=True)
    Gadv = tf.reduce_mean(Gadv)
    
    # 2. Classification loss
    
    Dcls = tf.keras.losses.binary_crossentropy(real_label, realcls, from_logits=True)
    Dcls = tf.reduce_mean(Dcls)
    
    Gcls = tf.keras.losses.binary_crossentropy(fake_label, fakecls, from_logits=True)
    Gcls = tf.reduce_mean(Gcls)
    
    # 3. Total loss
    
    Dloss = Dadv + (GAMMA/2)*R1 + LAMBDA_CLS*Dcls
    
    Gloss = Gadv + LAMBDA_CLS*Gcls
    
    return (Dloss, Dadv, Dcls, R1), (Gloss, Gadv, Gcls)

####################################################
#################### Training ######################
####################################################
def train_step(inp, mask, dopt, gopt):
    target_mask = tf.random.shuffle(mask)
    with tf.GradientTape(persistent=True) as tape:
        x2y = G([inp, target_mask])
        y2x = G([x2y, mask])
        
        (Dloss, Dadv, Dcls, R1),(Gloss, Gadv, Gcls) = model_loss(inp, x2y, mask, target_mask)
        rec_loss = recon_loss(inp, y2x)
        Gloss += LAMBDA_REC*rec_loss
    dgrads = tape.gradient(Dloss, D.trainable_variables)
    dopt.apply_gradients(zip(dgrads, D.trainable_variables))
    if gopt!=None:
        ggrads = tape.gradient(Gloss, G.trainable_variables)
        gopt.apply_gradients(zip(ggrads, G.trainable_variables))
    
    return (Dloss, Dadv, Dcls, R1), (Gloss, Gadv, Gcls, rec_loss)

def train(filelist,
          epochs=10,
          bs=BATCH_SIZE, 
          lr=(1e-4,1),
          gamma_decay=0.9,
          ln_decay=0,
          pic=1, 
          svwts=True):
    global GAMMA
    global LN
    train_step_graph = tf.function(train_step)
    # model_loss_graph = tf.function(model_loss)
    print("Starting training session lasting %d epochs"%(epochs))
    trtot = len(files)
    steps = trtot//bs + 1 if trtot%bs!=0 else trtot//bs
    
    gOpt = tf.keras.optimizers.Adam(lr[0], beta_1=0.5, beta_2=0.999)
    dOpt = tf.keras.optimizers.Adam(lr[0]/lr[1], beta_1=0.5, beta_2=0.999)
    for m in range(epochs):
        start_epoch = time.time()
        tots = np.zeros((8))
        for n in range(steps):
            start_split = time.time()
            first = n*bs;last = (n+1)*bs;amt = last-first
            
            inp = disk_to_tensor(filelist[first:last], PATH, RSZ)
            mask = Ann[first:last]
            
            arg = gOpt if n%N_CRITIC==0 else None
            (Dloss, Dadv, Dcls, R1), (Gloss, Gadv, Gcls, recon) = train_step_graph(inp, mask, dOpt, arg)
            
            for o,l in enumerate([Dloss, Gloss, Dadv, Gadv, Dcls, Gcls, recon, R1]):
                tots[o] += l.numpy()*amt
            
            sys.stdout.write("\rEpoch %d; Batch %d/%d; Losses (Dadv|Gadv): (%6.3f|%6.3f), (RClass|FClass): (%6.3f|%6.3f), L1Rec: %6.3f, R1: %6.3f; Batch time: %.1f"%(
                             m+1, n+1, steps, Dadv.numpy(), Gadv.numpy(), Dcls.numpy(), Gcls.numpy(), recon.numpy(), R1.numpy(), time.time()-start_split))

        sys.stdout.write("\rEpoch %d; GAMMA=%.2f; Total Loss (Dadv|Gadv): (%6.3f|%6.3f), (RClass|FClass): (%6.3f|%6.3f), L1Rec: %6.3f, R1: %6.3f; Time elapsed: %.1f%30s\n"%(
                         m+1, GAMMA, tots[2]/trtot, tots[3]/trtot, tots[4]/trtot, tots[5]/trtot, tots[6]/trtot, tots[7]/trtot, time.time()-start_epoch,''))
        GAMMA*=gamma_decay
        LN*=ln_decay
        if m>8:
            gOpt.learning_rate.assign(gOpt.learning_rate*0.9)
            dOpt.learning_rate.assign(dOpt.learning_rate*0.9)
        if (pic>0) & (m%pic==0):
            generate_images(FIXED_INP, fixed_ann, fixed_targ, fn='%d.jpg'%(m+1))
        if svwts:
            G.save_weights('C:/Users/glapi/Documents/Python/Tensorflow/MyProjects/celeba/StarGAN/weights/G.wts')
            D.save_weights('C:/Users/glapi/Documents/Python/Tensorflow/MyProjects/celeba/StarGAN/weights/D.wts')

def generate_images(image, orig_mask, targ_mask, fn='image.jpg'):
    if len(image.shape)<4:
        image = tf.expand_dims(image, axis=0)
    if len(orig_mask.shape)<2:
        orig_mask = tf.expand_dims(orig_mask, axis=0)
    if len(targ_mask.shape)<2:
        targ_mask = tf.expand_dims(targ_mask, axis=0)
    num = image.shape[0]
    
    output = G([image, targ_mask], training=False)
    orig_mask = orig_mask.numpy()
    targ_mask = targ_mask.numpy()
    output = output.numpy()*0.5+0.5
    image = image.numpy()*0.5+0.5
    fig = plt.figure(figsize=(20,20))
    for m in range(num):
        
        ax1 = plt.subplot(num, 2, 2*m+1)
        ax1.imshow(image[m])
        ax1.xaxis.set_ticks([-1])
        ax1.xaxis.set_ticklabels('')
        ax1.yaxis.set_ticks([-1])
        ax1.yaxis.set_ticklabels('')
        lab = "/".join(list(np.array(labels)[orig_mask[m]==True]))
        ax1.set_xlabel(lab, size=20)
        
        ax2 = plt.subplot(num, 2, 2*m+2)
        ax2.imshow(output[m])
        ax2.xaxis.set_ticks([-1])
        ax2.xaxis.set_ticklabels('')
        ax2.yaxis.set_ticks([-1])
        ax2.yaxis.set_ticklabels('')
        lab = "/".join(list(np.array(labels)[targ_mask[m]==True]))
        ax2.set_xlabel(lab, size=20)
    
    fig.savefig("C:/Users/glapi/Desktop/%s"%(fn))
    plt.close()
