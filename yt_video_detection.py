# Importing necessary packages
import subprocess

from customtkinter import *
from tkinter import *
import pytube
from pytube import YouTube
from tkinter import messagebox, filedialog
from test import *
import tkinter as tk
from PIL import Image, ImageTk


# Defining CreateWidgets() function
# to create necessary tkinter widgets
def Widgets():
    # Creating black canvas
    canvas = Canvas(root, bg="black", width=800, height=600)
    canvas.pack(fill=tk.BOTH, expand=tk.YES)
    #canvas.grid(row=0, column=0, sticky="nsew")

    # Creating frame
    frame = Frame(canvas, bg="black")
    #frame.pack(pady=20)
    #canvas.create_window((800,600), window=frame)
    frame_1 = Frame(canvas, bg="black")

    # Load the icon image - back
    icon_image = Image.open("icons/rotate-back.png")
    icon_image = icon_image.resize((45, 45))  # Resize the icon image to desired size
    icon_photo = ImageTk.PhotoImage(icon_image)

    # Load the icon image - test
    icon_image_1 = Image.open("icons/seo-search.png")
    icon_image_1 = icon_image_1.resize((45, 45))  # Resize the icon image to desired size
    icon_photo_1 = ImageTk.PhotoImage(icon_image_1)



    # Creating head label with updated style
    head_label = tk.Label(canvas, text="DETECT YOUTUBE DEEPFAKE VIDEOS",
                       padx=15,
                       pady=15,
                       font=("Roboto", 20)
                        , fg = "white", bg = "#000000"
                       )
    head_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)

    # Creating link label with updated style
    link_label = tk.Label(frame_1,
                       text="YOUTUBE LINK :",
                       pady=5,
                       padx=5,
                       font=("Roboto", 14)
                       , fg="white", bg="#000000")
    link_label.pack(side=tk.LEFT,  padx=(20, 0))

    # Creating entry for YouTube link with updated style
    root.linkText = CTkEntry(frame_1,
                          width=250,
                          textvariable=video_Link,
                          font=("Roboto", 14))

    root.linkText.pack(side=tk.RIGHT,  padx=(0, 20))

    # Creating download button with updated style
    test_B = CTkButton(frame,
                         text="Results",
                         command=test,
                         width=140,# Increased width to make it bigger
                         height=40,
                         fg_color="#FF0000",
                         font=("Roboto", 15), compound=tk.LEFT, image=icon_photo_1, hover_color="#FF0000")  # Increased font size
    test_B.pack(side=tk.RIGHT,  padx=(20, 0))

    # Creating back button with updated style
    back_B = CTkButton(frame,
                      text="HOME",
                      width=140,  # Increased width to match download button
                      height=40,
                      fg_color="#10ac84",  # Changed color to match download button
                      font=("Roboto", 15),compound=tk.LEFT, image=icon_photo, hover_color="#10ac84",
                       command=main_page)  # Increased font size to match download button
    back_B.pack(side=tk.LEFT,  padx=(20, 0))

    # Centering frame on canvas
    canvas.update_idletasks()  # Update canvas to get accurate frame size
    frame_width = frame.winfo_width()
    frame_height = frame.winfo_height()
    canvas.create_rectangle(0, 0, 520, 280, width=0)
    frame_1.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    frame.place(relx=0.5, rely=0.7, anchor=tk.CENTER)


"""
# Defining Browse() to select a
# destination folder to save the video

def Browse():
	# Presenting user with a pop-up for
	# directory selection. initialdir
	# argument is optional Retrieving the
	# user-input destination directory and
	# storing it in downloadDirectory
	download_Directory = filedialog.askdirectory(
		initialdir="YOUR DIRECTORY PATH", title="Save Video")

	# Displaying the directory in the directory
	# textbox
	download_Path.set(download_Directory)
"""


# Defining test() to display predictions
def test():
    YouTube(video_Link.get(), use_oauth=True, allow_oauth_cache=True).streams.filter(file_extension="mp4").first().download()

    file = [i for i in os.listdir(os.getcwd()) if i.endswith('.mp4')][0]
    faces = face_extractor.process_video(file)
    os.remove(file)
    fac = torch.stack([transf(image=frame['faces'][0])['image'] for frame in faces if len(frame['faces'])])

    with torch.no_grad():
        faces_pred = net(fac.to(device)).cpu().numpy().flatten()

    if expit(faces_pred.mean()) >= 0.5:
        print("Fake video :(")
        messagebox.showinfo("Result", "Fake video :(\n"+'Average score : {:.4f}'.format(expit(faces_pred.mean())))
    else:
        print("Real video !")
        messagebox.showinfo("Result", "Real video !\n"+'Average score : {:.4f}'.format(expit(faces_pred.mean())))
    print('Average score : {:.4f}'.format(expit(faces_pred.mean())))


def main_page():
    root.destroy()
    import gui


# Creating object of tk class
root = CTk(className="Deepfake videos detection for Youtube Videos")


# Setting the title, background color
# and size of the tkinter window and
# disabling the resizing property
root.geometry("520x280")
root.resizable(False, False)
root.title("YouTube Video Detection")
#root.config(background="PaleGreen1")

# Creating the tkinter Variables
video_Link = StringVar()
download_Path = StringVar()

# Calling the Widgets() function
Widgets()


# Defining infinite loop to run
# application
root.mainloop()
