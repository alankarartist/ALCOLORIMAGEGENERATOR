import numpy as np
import cv2
from tkinter import*
from tkinter import font
from PIL import ImageTk, Image
import os

cwd = os.path.dirname(os.path.realpath(__file__))

class AlColorImageGenerator:
    def __init__(self):
        root = Tk(className=" ALCOLORIMAGENERATOR ")
        root.geometry("400x130+1510+885")
        root.resizable(0,0)
        root.iconbitmap(os.path.join(cwd+'\\UI\\icons', 'alcolorimagegenerator.ico'))
        root.config(bg="#000000")
        root.overrideredirect(1)
        color = '#000000'
        
        def callback(event):
            root.geometry("400x130+1510+885")

        def showScreen(event):
            root.deiconify()
            root.overrideredirect(1)

        def screenAppear(event):
            root.overrideredirect(1)

        def hideScreen():
            root.overrideredirect(0)
            root.iconify()

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

        textHighlightFont = font.Font(family='OnePlus Sans Display', size=12)
        appHighlightFont = font.Font(family='OnePlus Sans Display', size=12, weight='bold')

        #title bar
        titleBar = Frame(root, bg='#141414', relief=SUNKEN, bd=0)
        icon = Image.open(os.path.join(cwd+'\\UI\\icons', 'alcolorimagegenerator.ico'))
        icon = icon.resize((30,30), Image.ANTIALIAS)
        icon = ImageTk.PhotoImage(icon)
        iconLabel = Label(titleBar, image=icon)
        iconLabel.photo = icon
        iconLabel.config(bg='#141414')
        iconLabel.grid(row=0,column=0,sticky="nsew")
        titleLabel = Label(titleBar, text='ALCOLORIMAGEGENERATOR', fg='#909090', bg='#141414', font=appHighlightFont)
        titleLabel.grid(row=0,column=1,sticky="nsew")
        closeButton = Button(titleBar, text="x", bg='#141414', fg="#909090", borderwidth=0, command=root.destroy, font=appHighlightFont)
        closeButton.grid(row=0,column=3,sticky="nsew")
        minimizeButton = Button(titleBar, text="-", bg='#141414', fg="#909090", borderwidth=0, command=hideScreen, font=appHighlightFont)
        minimizeButton.grid(row=0,column=2,sticky="nsew")
        titleBar.grid_columnconfigure(0,weight=1)
        titleBar.grid_columnconfigure(1,weight=20)
        titleBar.grid_columnconfigure(2,weight=1)
        titleBar.grid_columnconfigure(3,weight=1)
        titleBar.pack(fill=X)

        #image widget
        imageFile = Label(root, text="IMAGE TO BE COVERTED")
        imageFile.pack()
        imageFile.config(bg=color,fg="white",font=appHighlightFont)
        imageFile= Entry(root, bg="white", fg=color, highlightbackground=color, highlightcolor=color, highlightthickness=3, bd=0,font=textHighlightFont)
        imageFile.pack(fill=X)

        #submit button
        convert = Button(root, borderwidth=0, highlightthickness=5, text="CONVERT IMAGE", command=convert)
        convert.config(bg=color,fg="white",font=appHighlightFont)
        convert.pack(fill=X)

        titleBar.bind("<B1-Motion>", callback)
        titleBar.bind("<Button-3>", showScreen)
        titleBar.bind("<Map>", screenAppear)

        root.mainloop()

if __name__ == "__main__":
    AlColorImageGenerator()