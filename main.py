import numpy as np
from skimage.morphology import erosion, dilation, reconstruction
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.morphology import label
from skimage.segmentation import watershed
from skimage import color
from PIL import Image
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

# Definition de differents elements structurants
# Par defaut, l'origine est au centre
se1 = np.ones((15,15)) # square (boules 8-connexes)
se2 = square(15) # equivalent au precedent
se3 = np.ones((7)) # segment
se4 = diamond(3) # boules 4-connexes
se5 = disk(10) # boules euclidiennes


se6 = np.array([[0, 0, 1, 1, 1], # E.S. plat arbitraire
                [0, 1, 0, 0, 0,],
                [0, 1, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1]], dtype=np.uint8)
se7 = np.array([[7, 6, 6, 6, 7], # E.S. non plat
                [6, 5, 4, 5, 6],
                [6, 4, 0, 4, 6],
                [6, 5, 4, 5, 6],
                [7, 6, 6, 6, 7]], dtype=np.uint8)
se8 = square(5)
# Ouvrir une image en niveau de gris et conversion en tableau numpy
# au format uint8 (entier non signe entre 0 et 255)
img =  np.asarray(Image.open('./images/clock.png')).astype(np.uint8)

# Ouvrir une image binaire et conversion en tableau numpy
# au format booleen (code sur 0 et 1)
imgBin =  np.asarray(Image.open('./images/ghost.png')).astype(np.bool_)

################# Q1 ##################
imDil = dilation(img,se4) # Dilatation morphologique
imEro = erosion(img,se4) # Erosion morphologique
imgrai = img - imEro
imgrae = imDil - img
imo = dilation(imEro,se4)
imf = erosion(imDil,se4)
imgm = imDil - imEro
imlm = imgrae - imgrai
imth = img - imo
imthc = imf - img

# # Affichage avec matplotlib
plt.subplot(151)
plt.imshow(img,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Originale')
plt.subplot(152)
plt.imshow(imDil,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Dilatation')
plt.subplot(153)
plt.imshow(imEro,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Erosion')
plt.subplot(154)
plt.imshow(imo,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Ouverture')
plt.subplot(155)
plt.imshow(imf,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Fermeture')
plt.show()
plt.subplot(141)
plt.imshow(imgrai,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Gradient interne')
plt.subplot(142)
plt.imshow(imgrae,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Gradient externe')
plt.subplot(143)
plt.imshow(imgm,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Gradient morphologique')
plt.subplot(144)
plt.imshow(imlm,cmap = 'gray',vmin = -255.0,vmax = 255.0)
plt.title('Laplacien morphologique')
plt.show()
plt.subplot(121)
plt.imshow(imth,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Top-hat')
plt.subplot(122)
plt.imshow(imthc,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Top-hat conjugé')
plt.show()
# Fermer la fenetre pour passer aux traitements suivants

# experiment dual
imgA = erosion(255 - img,se4)
imgB = 255 - dilation(img, se4)
plt.subplot(121)
plt.imshow(imgA,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('ε(x*)')
plt.subplot(122)
plt.imshow(imgB,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('δ(x)*')
plt.show()

imgC = dilation(imgA,se4)
imgD = 255 - erosion(imEro,se4)
plt.subplot(121)
plt.imshow(imgA,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('γ(x*)')
plt.subplot(122)
plt.imshow(imgB,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('φ(x)*')
plt.show()


###################### Q2 #######################333

seuillage = 15
gammaplus = np.zeros((256,256))
gammaminus = np.zeros((256,256))
Gs = np.zeros((256,256))
for i in range(256):
    for j in range (256):
        if imgrae[i][j] > imgrai[i][j]:
            gammaplus[i][j] = 1
        else:
            gammaminus[i][j] = 1
        if imgm[i][j] >= seuillage:
            Gs[i][j] = 1
gammazero = dilation(gammaplus,diamond(1))
for i in range(256):
    for j in range (256):
        if Gs[i][j] == 1:
            if(gammazero[i][j])==0 | (gammaminus[i][j]==0):
                Gs[i][j] = 0
plt.subplot(121)
plt.imshow(img,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Original')
plt.subplot(122)
plt.imshow(Gs,cmap = 'binary')
plt.title('contour rudimentaire')
plt.show()

############################Q3#########################
H = diamond(2)
M = diamond(2)
for j in range(3, 5):
    for i in range(5):
        H[i][j] = 0
for j in range(3):
    for i in range(5):
        M[i][j] = 0

# H = square(7)
# M = square(7)
# for j in range(4, 7):
#     for i in range(7):
#         H[i][j] = 0
# for j in range(4):
#     for i in range(7):
#         M[i][j] = 0

imginverted = (imgBin == False)
imgeroH = erosion(imgBin,H)
imgeroM = erosion(imginverted,M)
img_tout_ou_rien = imgeroH & imgeroM

plt.subplot(131)
plt.imshow(imgeroH,cmap = 'binary')
plt.title('tout-ou-rien')
plt.subplot(132)
plt.imshow(imgeroM,cmap = 'binary')
plt.title('tout-ou-rien')
plt.subplot(133)
plt.imshow(img_tout_ou_rien,cmap = 'binary')
plt.title('tout-ou-rien')
plt.show()

######################## Q4.1##########################

def RECONSTRUIT (img,img_ini):
    size = np.size(img[0])
    imtmp = np.tile(img,1)
    for k in range(20):  ### répéter jusqu'à stable
            imtmp = dilation(imtmp,disk(2))
            for i in range(size):
                for j in range(size):
                    imtmp[i][j] = min(img_ini[i][j], imtmp[i][j])
    return imtmp

######################## Q4.2##########################
img =  np.asarray(Image.open('./images/particules.png')).astype(np.uint8)
imEro42 = erosion(img,disk(10))
imr = RECONSTRUIT(imEro42, img)

plt.subplot(131)
plt.imshow(img==False,cmap = 'binary')
plt.title('Originale')
plt.subplot(132)
plt.imshow(imEro42==False,cmap = 'binary')
plt.title('Erosion (disque euclidien)')
plt.subplot(133)
plt.imshow(imr==False,cmap = 'binary')
plt.title('Reconstruction')
plt.show()

img =  np.asarray(Image.open('./images/tree_noise.png')).astype(np.uint8)
size = np.size(img[0])
imEro43 = erosion(img,disk(3))
imr = RECONSTRUIT(imEro43, img)


# Affichage avec matplotlib
plt.subplot(131)
plt.imshow(img,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Originale')
plt.subplot(132)
plt.imshow(imr,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Ouverture par reconstruction')

imDil43 = dilation(img,disk(3))
imr = reconstruction(imDil43,img,method='erosion')

plt.subplot(133)
plt.imshow(imr,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Fermeture par reconstruction')
plt.show()

######################## Q4.3 ##########################
img =  np.asarray(Image.open('./images/coffee.png')).astype(np.uint8)

def EROSION_ULTIME(img):
    imX = np.tile(img,1)
    size = np.size(img[0])
    imEroU = np.zeros((size,size),dtype = int)
    for k in range(10): ##répéter jusqu'à stable
        imtmp = np.tile(imX,1)
        imX = erosion(imX, diamond(2))
        imY = RECONSTRUIT(imX, imtmp)
        Utmp = imtmp - imY
        for i in range(size):
            for j in range(size):
                if Utmp[i][j] == 1:
                    imEroU [i][j] = 1
    return imEroU

imEroU = EROSION_ULTIME(img)
plt.subplot(121)
plt.imshow(img==False,cmap = 'binary')
plt.title('Original')
plt.subplot(122)
plt.imshow(imEroU==False,cmap = 'binary')
plt.title('Érosion ultime')
plt.show()


######################## Q5.1 ##########################

# Ouvrir une image en niveau de gris et conversion en tableau numpy
# au format uint8 (entier non signe entre 0 et 255)
img_ui =  np.asarray(Image.open('./images/uranium.png')).astype(np.uint8)

## uncommenter pour une image naturelle
# img_ui =  np.asarray(Image.open('./images/goldhill.png')).astype(np.uint8)

####################### Q5.2 filtre spatial ######################

#imgradient = dilation(img_ui, disk(2)) - erosion(img_ui, disk(2))
#im_rec = reconstruction(dilation(imgradient,disk(2)),imgradient,method='erosion')

#img_rec = reconstruction(255 - im_rec,255 - im_rec + 1, selem=diamond(1))
#ui_min_reg = (img_rec != 255 - im_rec + 1)


###################### Q5.3 filtre dynamique ######################
# img_rec = reconstruction(255 - img_ui,255 - img_ui + 20, selem=diamond(1))
# ui_min_reg = (img_rec != 255 - img_ui + 20)


##################### Q5.4 filtre conbine ############################
imgradient = dilation(img_ui, disk(2)) - erosion(img_ui, disk(2))
im_rec = reconstruction(dilation(imgradient,disk(2)),imgradient,method='erosion')
img_rec = reconstruction(255 - im_rec,255 - im_rec + 20, selem=diamond(1)) ## ici h = 20
ui_min_reg = (img_rec != 255 - im_rec + 20)

seeds = label(ui_min_reg,neighbors = 8) # Etiquetage des minima
ui_ws = watershed(img_ui,seeds) # LPE par defaut : Marqueur = Minima Regionaux
ws_display=color.label2rgb(ui_ws,img_ui)

# Affichage avec matplotlib
plt.subplot(131)
plt.imshow(img_ui,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Originale')
plt.subplot(132)
plt.imshow(ui_min_reg,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Minima regionaux')
plt.subplot(133)
plt.imshow(ws_display)
plt.title('Ligne de Partage des Eaux')
plt.show()


############################ Q6 ##################
img =  np.asarray(Image.open('./images/Ingres.jpg')).astype(np.uint8)
def Contraste(image ,type):
    str_size = int(input("insérer la taille de l'élément structurant\n"))
    str = []
    correct = False
    while(correct==False):
        str_type = input("choisir un type de de l'élément structurant parmi disk, diamond et square\n")
        if str_type == 'disk':
            str = disk(str_size)
            correct = True
        elif str_type == 'square':
            str = square(str_size)
            correct = True
        elif str_type == 'diamond':
            str = diamond(str_size)
            correct = True

    size1 = image.shape[0]
    size2 = image.shape[1]
    imconstrast = np.zeros((size1, size2))
    if type == 0:
        im1 = erosion(image,str)
        im2 = dilation(image,str)
    elif type == 1:
        im1 = dilation(erosion(image,str),str)
        im2 = erosion(dilation(image,str),str)
    elif type == 2:
        im1 = reconstruction(erosion(image,str),image)
        im2 = reconstruction(dilation(image,str),image,method = 'erosion')
    else :
        return "please choose type from 0 1 and 2"

    for i in range(size1):
        for j in range(size2):
            if abs(im1[i][j] - image[i][j]) < abs(im2[i][j] - image[i][j]):
                imconstrast[i][j] = im1[i][j]
            else:
                imconstrast[i][j] = im2[i][j]
    return imconstrast

imgcon = Contraste(img,0)

plt.subplot(121)
plt.imshow(img, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('Originale')
plt.subplot(122)
plt.imshow(imgcon, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('image rehaussée')

plt.show()

######################## Q7 #####################
img =  np.asarray(Image.open('./images/tree_noise.png')).astype(np.uint8)
def FAS(image, premier, rayon):
    imtmp = np.tile(image,1)
    if premier == 0:
        for i in range(rayon):
            str = disk(i)
            imtmp = dilation(erosion(imtmp, str),str)
            imtmp = erosion(dilation(imtmp,str),str)

    elif premier == 1:
        for i in range(rayon):
            str = disk(i)
            imtmp = erosion(dilation(imtmp, str), str)
            imtmp = dilation(erosion(imtmp, str), str)
    else:
        return 'premier should be 0 or 1'
    return imtmp

def Nivellement(image,image_ref,cx):
    if cx != 4 | cx != 8:
        return "cx should be 4 or 8"
    size1 = image.shape[0]
    size2 = image.shape[1]
    sup = np.zeros((size1, size2)) ##image supérieure
    inf = np.zeros((size1, size2)) ##image inférieure
    indicateur = np.zeros((size1, size2))  ##indicateur
    result = np.zeros((size1, size2))
    for i in range(size1):
        for j in range(size2):
            if image[i][j] < image_ref[i][j]:
                sup[i][j] = 255
                inf[i][j] = image[i][j]
                indicateur[i][j] = 1
            else:
                sup[i][j] = image[i][j]
                inf[i][j] = 0
                indicateur[i][j] = 0
    recon_inf = reconstruction(inf,image_ref)
    recon_sup = 255 - reconstruction(255-sup,255-image_ref)
    for i in range(size1):
        for j in range(size2):
            if indicateur[i][j] == 1:
                result[i][j] = recon_inf[i][j]
            else:
                result[i][j] = recon_sup[i][j]
    return result


img_1 = FAS(img, 0, 1)
img_2 = FAS(img, 0, 2)
img_3 = FAS(img, 0, 3)
img_4 = FAS(img, 0, 4)
img_5 = FAS(img, 0, 5)

# img_1 = FAS(img, 1, 1)
# img_2 = FAS(img, 1, 2)
# img_3 = FAS(img, 1, 3)
# img_4 = FAS(img, 1, 4)
# img_5 = FAS(img, 1, 5)

niv1 = Nivellement(img_1, img, 4)
niv2 = Nivellement(img_2, img, 4)
niv3 = Nivellement(img_3, img, 4)
niv4 = Nivellement(img_4, img, 4)
niv5 = Nivellement(img_5, img, 4)


plt.subplot(131)
plt.imshow(img, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('Originale')
plt.subplot(132)
plt.imshow(img_1, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('FAS_0 rayon=1')
plt.subplot(133)
plt.imshow(img_2, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('FAS_0 rayon=2')
plt.show()

plt.subplot(131)
plt.imshow(img_3, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('FAS_0 rayon=3')
plt.subplot(132)
plt.imshow(img_4, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('FAS_0 rayon=4')
plt.subplot(133)
plt.imshow(img_5, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('FAS_0 rayon=5')
plt.show()

plt.subplot(131)
plt.imshow(img, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('Originale')
plt.subplot(132)
plt.imshow(niv1, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('niv_1 rayon=1')
plt.subplot(133)
plt.imshow(niv2, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('niv_2 rayon=2')
plt.show()

plt.subplot(131)
plt.imshow(niv3, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('niv_3 rayon=3')
plt.subplot(132)
plt.imshow(niv4, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('niv_4 rayon=4')
plt.subplot(133)
plt.imshow(niv5, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('niv_5 rayon=5')
plt.show()