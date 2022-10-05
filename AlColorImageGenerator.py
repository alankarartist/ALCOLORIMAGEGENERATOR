import numpy as np
import cv2
from tkinter import Tk, END, SUNKEN, Label, Button, X
from tkinter import font, filedialog, Text, BOTH, Frame
from PIL import ImageTk, Image
import os
import platform

cwd = os.path.dirname(os.path.realpath(__file__))
systemName = platform.system()


class AlColorImageGenerator:
    def __init__(self):
        root = Tk(className=" ALCOLORIMAGENERATOR ")
        root.geometry("400x150+1510+865")
        root.resizable(0, 0)
        iconPath = os.path.join(cwd+'\\UI\\icons',
                                'alcolorimagegenerator.ico')
        if systemName == 'Darwin':
            iconPath = iconPath.replace('\\','/')
        root.iconbitmap(iconPath)
        root.config(bg="#000000")
        root.overrideredirect(1)
        color = '#000000'

        def liftWindow():
            root.lift()
            root.after(1000, liftWindow)

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

        def openImage():
            imageFileEntry.delete(1.0, END)
            filename = filedialog.askopenfilename(filetypes=[('Image Files',
                                                              '*.jpg *.jpeg ' +
                                                              '*.bmp *.png ' +
                                                              '*.webp *.tiff')]
                                                  )
            imageFileEntry.insert(1.0, filename)

        def convert():
            ptPath = (cwd+'\\AlColorImageGenerator\\model'
                      '\\colorization_deploy_v2.prototxt')
            modelPath = (cwd+'\\AlColorImageGenerator\\model'
                         '\\colorization_release_v2' +
                         '.caffemodel')
            npyPath = (cwd+'\\AlColorImageGenerator\\model\\'
                       'pts_in_hull.npy')
            if systemName == 'Darwin':
                ptPath = ptPath.replace('\\','/')
                modelPath = modelPath.replace('\\','/')
                npyPath = npyPath.replace('\\','/')
            net = cv2.dnn.readNetFromCaffe(ptPath, modelPath)
            pts = np.load(npyPath)
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            pts = pts.transpose().reshape(2, 313, 1, 1)
            net.getLayer(class8).blobs = [pts.astype("float32")]
            net.getLayer(conv8).blobs = [np.full([1, 313], 2.606,
                                                 dtype='float32')]
            img = imageFileEntry.get("1.0", END)
            img = img.replace('/', '\\')[:-1]
            if systemName == 'Darwin':
                img = img.replace('\\','/')
            nimg = os.path.basename(img)
            cimg = 'color_' + nimg
            image = cv2.imread(img)
            scaled = image.astype("float32")/255.0
            lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50
            net.setInput(cv2.dnn.blobFromImage(L))
            ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized, 0, 1)
            colorized = (255 * colorized).astype("uint8")
            cimg = os.path.join(cwd+'\\AlColorImageGenerator\\images\\color',
                                cimg)
            if systemName == 'Darwin':
                cimg = cimg.replace('\\','/')
            cv2.imwrite(cimg, colorized)
            cv2.imshow("Original", image)
            cv2.imshow("Colorized", colorized)
            cv2.waitKey(0)

        textHighlightFont = font.Font(family='OnePlus Sans Display', size=12)
        appHighlightFont = font.Font(family='OnePlus Sans Display', size=12,
                                     weight='bold')

        titleBar = Frame(root, bg='#141414', relief=SUNKEN, bd=0)
        icon = Image.open(iconPath)
        icon = icon.resize((30, 30), Image.ANTIALIAS)
        icon = ImageTk.PhotoImage(icon)
        iconLabel = Label(titleBar, image=icon)
        iconLabel.photo = icon
        iconLabel.config(bg='#141414')
        iconLabel.grid(row=0, column=0, sticky="nsew")
        titleLabel = Label(titleBar, text='ALCOLORIMAGEGENERATOR',
                           fg='#909090', bg='#141414', font=appHighlightFont)
        titleLabel.grid(row=0, column=1, sticky="nsew")
        closeButton = Button(titleBar, text="x", bg='#141414', fg="#909090",
                             borderwidth=0, command=root.destroy,
                             font=appHighlightFont)
        closeButton.grid(row=0, column=3, sticky="nsew")
        minimizeButton = Button(titleBar, text="-", bg='#141414', fg="#909090",
                                borderwidth=0, command=hideScreen,
                                font=appHighlightFont)
        minimizeButton.grid(row=0, column=2, sticky="nsew")
        titleBar.grid_columnconfigure(0, weight=1)
        titleBar.grid_columnconfigure(1, weight=20)
        titleBar.grid_columnconfigure(2, weight=1)
        titleBar.grid_columnconfigure(3, weight=1)
        titleBar.pack(fill=X)

        imageFile = Button(root, text="IMAGE TO BE COVERTED", borderwidth=0,
                           highlightthickness=3, command=openImage)
        imageFile.pack(fill=X)
        imageFile.config(bg=color, fg="white", font=appHighlightFont)
        imageFileEntry = Text(root, bg="white", fg=color,
                              highlightbackground=color, highlightcolor=color,
                              highlightthickness=3, bd=0,
                              font=textHighlightFont, height=1)
        imageFileEntry.pack(fill=BOTH, expand=True)

        convert = Button(root, borderwidth=0, highlightthickness=5,
                         text="CONVERT IMAGE", command=convert)
        convert.config(bg=color, fg="white", font=appHighlightFont)
        convert.pack(fill=X)

        titleBar.bind("<B1-Motion>", callback)
        titleBar.bind("<Button-3>", showScreen)
        titleBar.bind("<Map>", screenAppear)

        if systemName == 'Windows':
            liftWindow()
        root.mainloop()


if __name__ == "__main__":
    AlColorImageGenerator()
