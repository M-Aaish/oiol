import numpy as np
import cv2
import math
import random

def rn():
    return random.random()

brushes = {}

# load brushes from ./brushes directory
def load_brushes():
    brush_dir = './shape_gen/brushes/'
    import os
    for fn in os.listdir(brush_dir):
        if os.path.isfile(brush_dir + fn):
            brush = cv2.imread(brush_dir + fn,0)
            if not brush is None:
                brushes[fn] = brush

load_brushes()

def get_brush(key='random'):
    if key=='random':
        key = random.choice(list(brushes.keys()))
    brush = brushes[key]
    return brush,key

def rotate_brush(brush,rad,srad,angle):
    # brush image should be of grayscale, pointing upwards

    # translate w x h into an area of 2rad x 2rad

    bh,bw = brush.shape[0:2]
    # print(brush.shape)

    osf = 0.1
    # oversizefactor: ratio of dist-to-edge to width,
    # to compensate for the patch smaller than the original ellipse

    rad = int(rad*(1.+osf))
    srad = int(srad*(1.+osf))

    # 1. scale
    orig_points = np.array([[bw/2,0],[0,bh/2],[bw,bh/2]]).astype('float32')
    # x,y of top left right
    translated = np.array([[rad,0],[rad-srad,rad],[rad+srad,rad]]).astype('float32')

    # affine transform matrix
    at = cv2.getAffineTransform(orig_points,translated)

    at = np.vstack([at,[0,0,1.]])

    # 2. rotate
    rm = cv2.getRotationMatrix2D((rad,rad),angle-90,1)
    # per document:
    # angle – Rotation angle in degrees. Positive values mean counter-clockwise rotation (the coordinate origin is assumed to be the top-left corner).

    # stroke image should point eastwards for 0 deg, hence the -90

    rm = np.vstack([rm,[0,0,1.]])

    # 3. combine 2 affine transform
    cb = np.dot(rm,at)
    # print(cb)

    # 4. do the transform
    res = cv2.warpAffine(brush,cb[0:2,:],(rad*2,rad*2))
    return res

def lc(i): #low clip
    return int(max(0,i))

def generate_motion_blur_kernel(dim=3,angle=0.,threshold_factor=1.3,divide_by_dim=True):
    radian = angle/360*math.pi*2 + math.pi/2
    # perpendicular
    # x2,y2 = math.cos(radian),math.sin(radian)
    # the directional vector

    # first, generate xslope and yslope
    gby,gbx = np.mgrid[0:dim,0:dim]
    cen = (dim+1)/2-1
    gbx = gbx - float(cen)
    gby = gby - float(cen)
    # gbx = gbx / float(cen) -.5
    # gby = gby / float(cen) -.5

    # then mix the slopes according to angle
    gbmix = gbx * math.cos(radian) - gby * math.sin(radian)

    kernel = (threshold_factor - gbmix*gbmix).clip(min=0.,max=1.)

    # center = (dim+1)/2-1
    # kernel = np.zeros((dim,dim)).astype('float32')

    # # iterate thru kernel
    # ki = np.nditer(kernel,op_flags=['readwrite'],flags=['multi_index'])
    # while not ki.finished:
    #     ya,xa = ki.multi_index
    #     y1,x1 = -(ya-center),xa-center # flip y since it's image axis
    #
    #     # now y1, x1 correspond to each kernel pixel's cartesian coord
    #     # with center being 0,0
    #
    #     dotp = x1*x2 + y1*y2 # dotprod
    #
    #     ki[0] = dotp
    #     ki.iternext()
    #
    # kernel = (threshold_factor-kernel*kernel).clip(min=0,max=1)

    if divide_by_dim:
        kernel /= dim * dim * np.mean(kernel)
    else:
        pass

    return kernel.astype('float32')

from colormixer import oilpaint_converters
b2p,p2b = oilpaint_converters()

def sigmoid_array(x):
    sgm = 1. / (1. + np.exp(-x))
    return np.clip(sgm * 1.2 - .1,a_max=1.,a_min=0.)

# the brush process
def compose(orig,brush,x,y,rad,srad,angle,color,usefloat=False,useoil=False,lock=None):
    # generate, scale and rotate the brush as needed
    brush_image = rotated = rotate_brush(brush,rad,srad,angle) # as alpha
    brush_image = np.reshape(brush_image,brush_image.shape+(1,)) # cast alpha into (h,w,1)

    if useoil:
        # gradient based color mixing
        # generate a blend map `gbmix` that blend in the direction of the brush stroke

        # first, generate xslope and yslope
        gby,gbx = np.mgrid[0:brush_image.shape[0],0:brush_image.shape[1]]
        gbx = gbx / float(brush_image.shape[1]) -.5
        gby = gby / float(brush_image.shape[0]) -.5

        dgx,dgy = rn()-.5,rn()-.5 # variation to angle
        # then mix the slopes according to angle
        gbmix = gbx * math.cos(angle/180.*math.pi+dgx) - gby * math.sin(angle/180.*math.pi+dgx)

        # some noise?
        gbmix += np.random.normal(loc=0.15,scale=.2,size=gbmix.shape)

        #strenthen the slope
        gbmix = sigmoid_array(gbmix*10)
        gbmix = np.reshape(gbmix,gbmix.shape+(1,)).astype('float32')

        # cv2.imshow('gbmix',gbmix)
        # cv2.waitKey(1)

    # width and height of brush image
    bh = brush_image.shape[0]
    bw = brush_image.shape[1]

    y,x = int(y),int(x)

    # calculate roi params within orig to paint the brush
    ym,yp,xm,xp = y-bh/2,y+bh/2,x-bw/2,x+bw/2

    # w and h of orig
    orig_h,orig_w = orig.shape[0:2]

    #crop the brush if exceed orig or <0
    alpha = brush_image[lc(0-ym):lc(bh-(yp-orig_h)),lc(0-xm):lc(bw-(xp-orig_w))]
    if useoil:
        gbmix = gbmix[lc(0-ym):lc(bh-(yp-orig_h)),lc(0-xm):lc(bw-(xp-orig_w))]

    #crop the roi params if < 0
    ym,yp,xm,xp = lc(ym),lc(yp),lc(xm),lc(xp)

    # print(alpha.shape,roi.shape)

    if alpha.shape[0]==0 or alpha.shape[1]==0: #or roi.shape[0]==0 or roi.shape[1]==0:
        # optimization: assume roi is valid

        print('alert: compose got empty roi')
        # dont paint if roi or brush is empty
    else:

        # to simulate oil painting mixing:
        # color should blend in some fasion from given color to bg color
        if useoil:
            if usefloat: # to 0,1
                pass
            else:
                # roi = roi.astype('float32')/255.
                color = np.array(color).astype('float32')/255.

            alpha = alpha.astype('float32')/255.

            # convert into oilpaint space
            color = b2p(color)

            def getkernelsize(r):
                k = min(55,int(r))
                if k%2==0:
                    k+=1
                if k<3:
                    k+=2
                return k
            sdim = getkernelsize(srad/5) # determine the blur kernel characteristics
            ldim = getkernelsize(rad/5)
            ssdim = getkernelsize(srad/7)

            #blur brush pattern
            softalpha = cv2.blur(alpha,(sdim,sdim)) # 0-1

            mixing_ratio = np.random.rand(alpha.shape[0],alpha.shape[1],1)
            # random [0,1] within shape (h,w,1

            # increase mixing_ratio where brush pattern
            # density is lower than 1
            # i.e. edge enhance
            mixing_ratio[:,:,0] += (1-softalpha)*2

            mixing_th = 0.1 # threshold, larger => mix more
            mixing_ratio = mixing_ratio > mixing_th
            # threshold into [0,1]

            # note: mixing_ratio is of dtype bool

            # apply motion blur on the mixed colormap
            kern = generate_motion_blur_kernel(dim=ldim,angle=angle)

            # print(sdim,ldim,kern.shape,colormap.dtype,kern.dtype,mixing_ratio.dtype,roi.dtype,color.dtype)

            # sample randomly from roi
            # random is acturally bad idea
            # limit sample area under alpha is better
            ry,rx = 0,0
            n = 20
            while n>0:
                ry,rx = int(rn()*alpha.shape[0]),int(rn()*alpha.shape[1])
                if alpha[ry,rx]>.5:
                    break
                n-=1

            # roi loading moved downwards, for optimization
            roi = orig[ym:yp,xm:xp]

            if usefloat: # roi to 0,1
                pass
            else:
                roi = roi.astype('float32')/255.

            roi = b2p(roi)

            if n>0:
                random_color = roi[ry,rx]
                tipcolor = color * gbmix + random_color  * (1 - gbmix)
            else:
                tipcolor = color

            # add some gaussian to color maybe?
            # remember: color should be in UABS space
            # tipcolor = np.random.normal(loc=color,scale=.04,size=alpha.shape[0:2]+(3,))
            # mixing_ratio2 = np.random.rand(alpha.shape[0],alpha.shape[1],1)

            # mixing_ratio2 = np.mgrid[]

            # mixing_ratio2 = mixing_ratio2 > 0.4

            #exp: how bout tip color blend from 1 color to another..

            # tipcolor = np.clip(tipcolor,a_max = 1.-1e-6, a_min = 1e-6)

            # tipcolor = cv2.filter2D(tipcolor,cv2.CV_32F,cv2.blur(kern,(sdim,sdim)))
            # tipcolor = cv2.blur(tipcolor,(sdim,sdim))

            # blend tip color (image) with bg
            # larger the mixing_ratio, stronger the (tip) color
            ia = (1 - mixing_ratio).astype('float32')
            ca = tipcolor * mixing_ratio
            # print(roi.dtype,ia.dtype,ca.dtype)
            colormap = roi * ia + ca
            # print(colormap.dtype,kern.dtype)

            #mblur
            colormap = cv2.filter2D(colormap,cv2.CV_32F,kern)

            ########

            #final composition
            ca = colormap*alpha
            ia = 1-alpha

            if lock is not None:
                lock.acquire()
                # print('lock acquired for brush @',x,y)
                # if canvas lock provided, acquire it. this prevents overwrite problems

            #final loading of roi.
            roi=orig[ym:yp,xm:xp]

            if usefloat:
                roi = b2p(roi)
                orig[ym:yp,xm:xp] = p2b(roi*ia+ca)
            else:
                roi = b2p(roi.astype('float32')/255.)
                orig[ym:yp,xm:xp] = p2b(roi*ia+ca)*255.
        else:
            # no oil painting
            colormap = np.array(color).astype('float32') # don't blend with bg, just paint fg

            if usefloat:
                alpha = alpha.astype('float32')/255.
                ia = 1-alpha
                ca = colormap*alpha
            else:
                # integer version
                colormap = colormap.astype('uint32')
                ia = 255-alpha
                ca = colormap*alpha

            if lock is not None:
                lock.acquire()
                # print('lock acquired for brush @',x,y)
                # if canvas lock provided, acquire it. this prevents overwrite problems

            roi = orig[ym:yp,xm:xp]

            if usefloat:
                # if original image is float
                orig[ym:yp,xm:xp] = roi * ia + ca
            else:
                roi = roi.astype('uint32')
                # use uint32 to prevent multiplication overflow
                orig[ym:yp,xm:xp] = (roi * ia + ca)/255

    # painted
    if lock is not None:
        lock.release()
        # print('lock released for brush @',x,y)
        # if canvas lock provided, release it. this prevents overwrite problems

def test(onlyfloat=False,onlyoil=False):
    flower = cv2.imread('flower.jpg')
    if not onlyfloat:
        fint = flower.copy()
        for i in range(100):
            brush,key = get_brush()
            color = [rn()*255,rn()*255,rn()*255]

            if not onlyoil:
                print('integer no oil')
                compose(fint,brush,x=rn()*flower.shape[1],y=rn()*flower.shape[0],
                rad=50,srad=10+20*rn(),angle=rn()*360,color=color,useoil=False)

            print('integer oil')
            compose(fint,brush,x=rn()*flower.shape[1],y=rn()*flower.shape[0],
            rad=50,srad=10+20*rn(),angle=rn()*360,color=color,useoil=True)

            cv2.imshow('integer',fint)
            cv2.waitKey(10)

    floaty = flower.copy().astype('float32')/255.
    for i in range(100):
        brush,key = get_brush()
        color = [rn(),rn(),rn()]

        if not onlyoil:
            print('float no oil')
            compose(floaty,brush,x=rn()*flower.shape[1],y=rn()*flower.shape[0],
            rad=50,srad=10+20*rn(),angle=rn()*360,color=color,usefloat=True,useoil=False)

        print('float oil')
        compose(floaty,brush,x=rn()*flower.shape[1],y=rn()*flower.shape[0],
        rad=50,srad=10+20*rn(),angle=rn()*360,color=color,usefloat=True,useoil=True)

        cv2.imshow('float',floaty)
        cv2.waitKey(10)
