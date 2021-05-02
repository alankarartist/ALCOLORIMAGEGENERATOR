import numpy as np
import cv2
from tkinter import*
from tkinter import font
import os

cwd = os.path.dirname(os.path.realpath(__file__))

class AlColorImageGenerator:
    def __init__(self):
        root = Tk(className=" ALCOLORIMAGENERATOR ")
        root.geometry("350x90+1560+925")
        root.resizable(0,0)
        root.iconbitmap(os.path.join(cwd+'\\UI\\icons', 'alcolorimagegenerator.ico'))
        root.config(bg="#FFFFFF")
        color = '#FFFFFF'

        def convert():
            net = cv2.dnn.readNetFromCaffe(cwd+'\AlColorImageGenerator\model\colorization_deploy_v2.prototxt',cwd+'\AlColorImageGenerator\model\colorization_release_v2.caffemodel')
            pts = np.load(cwd+'\AlColorImageGenerator\model\pts_in_hull.npy')
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            pts = pts.transpose().reshape(2,313,1,1)
            net.getLayer(class8).blobs = [pts.astype("float32")]
            net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]
            img = imageFile.get()
            cimg = 'color_' + img
            img = os.path.join(cwd+'\AlColorImageGenerator\images', 'bw\\'+img)
            print(img)
            image = cv2.imread(img)
            scaled = image.astype("float32")/255.0
            lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)
            resized = cv2.resize(lab,(224,224))
            L = cv2.split(resized)[0]
            L -= 50
            net.setInput(cv2.dnn.blobFromImage(L))
            ab = net.forward()[0, :, :, :].transpose((1,2,0))
            ab = cv2.resize(ab, (image.shape[1],image.shape[0]))
            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)
            colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized,0,1)
            colorized = (255 * colorized).astype("uint8")
            cimg = os.path.join(cwd+'\AlColorImageGenerator\images\color', cimg)
            cv2.imwrite(cimg, colorized) 
            cv2.imshow("Original",image)
            cv2.imshow("Colorized",colorized)
            cv2.waitKey(0)

        appHighlightFont = font.Font(family='sans-serif', size=12, weight='bold')
        textHighlightFont = font.Font(family='Segoe UI', size=12, weight='bold')

        #image widget
        imageFile = Label(root, text="IMAGE TO BE COVERTED")
        imageFile.pack()
        imageFile.config(bg=color,fg="black",font=textHighlightFont)
        imageFile= Entry(root, bg="black", fg=color, highlightbackground=color, highlightcolor=color, highlightthickness=3, bd=0,font=appHighlightFont)
        imageFile.pack(fill=X)

        #submit button
        convert = Button(root, borderwidth=0, highlightthickness=5, text="CONVERT IMAGE", command=convert)
        convert.config(bg=color,fg="black",font=textHighlightFont)
        convert.pack(fill=X)

        root.mainloop()

if __name__ == "__main__":
    AlColorImageGenerator()