import tkinter as tk
from tkinter import *
import customtkinter as ct
from PIL import Image, ImageTk
import pytube
from pytube import *
import subprocess
from pytube import YouTube

# import filedialog module
from tkinter import filedialog
import tkinter.messagebox as messagebox
from test import *

# Function to handle button click
def open_file_dialog():
    global file_path
    file_path = filedialog.askopenfilename(title="Please select a Video")
    if file_path:
        display_video(file_path)
        upload_button.pack_forget()  # Hide the "Upload Video" button
        youtube_button.pack_forget()  # Hide the "Youtube button" button
        restart_button.pack(pady=10)  # Show the "Restart" button
        view_results_button.pack(pady=10)  # Show the "View Results" button
        video_label.configure(text="")
        frame.place(relx=0.5, rely=0.7, anchor=tk.CENTER)
        root.geometry("800x800")


# Function to restart video upload process
def restart_upload():
    cap.release()  # Release the video capture object
    video_label.configure(image=None)  # Clear the video label
    restart_button.pack_forget()  # Hide the "Restart" button
    view_results_button.pack_forget()  # Hide the "View Results" button
    upload_button.pack()  # Show the "Upload Video" button


# Function to display video on GUI
def display_video(file_path):
    global cap
    # Release the video capture object if it exists
    if cap is not None:
        cap.release()
    # Open video capture object
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set fixed video frame size
    frame_width = 300
    frame_height = 300

    # Create label with fixed size
    video_label.configure(width=frame_width, height=frame_height)

    # Function to update video frame on label
    def update_video_frame():
        ret, frame = cap.read()
        if ret:
            # Resize frame to fixed size
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Convert frame to image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Convert image to PhotoImage and update label
            tk_image = ImageTk.PhotoImage(image=pil_image)
            video_label.configure(image=tk_image)
            video_label.image = tk_image

            # Call update_video_frame() after delay
            video_label.after(fps, update_video_frame)

    update_video_frame()

def go_back():
 # Destroy the current GUI
    root.destroy()
    import gui
    # Open the new GUI
    # subprocess.Popen(['python', 'gui.py'])

def youtube_page():
    root.destroy()
    import yt_video_detection

# Create main window
root = ct.CTk(className="Deepfake videos detection")
#root.configure(bg="blue")
root.title("Deepfake Videos Detection")
# Set window size
root.geometry("500x400")

#root.resizable(False, False)

# Create a canvas that spans the entire window
canvas = tk.Canvas(root, width=800, height=600, bg="black")
canvas.pack(fill=tk.BOTH, expand=tk.YES)

# Create a frame to hold the buttons results and restart
frame = Frame(canvas, bg="black")
frame.pack_forget()


# Create label to display video
video_label = Label(canvas, width=250, font=("Roboto", 20), fg="white", bg="#000000")
video_label.configure(text="DEEPFAKE VIDEOS DETECTION")
video_label.pack(pady=20)

# Load the icon image - youtube
icon_image = Image.open("icons/social-video-youtube-clip.png")
icon_image = icon_image.resize((45, 45))  # Resize the icon image to desired size
icon_photo = ImageTk.PhotoImage(icon_image)

# Load the icon image - upload
icon_image = Image.open("icons/upload-button-alternate.png")
icon_image = icon_image.resize((50, 50))  # Resize the icon image to desired size
icon_photo_1 = ImageTk.PhotoImage(icon_image)

# Load the icon image - Back home
icon_image = Image.open("icons/house-chimney-1-alternate.png")
icon_image = icon_image.resize((30, 30))  # Resize the icon image to desired size
icon_photo_2 = ImageTk.PhotoImage(icon_image)

# Load the icon image - Restart
icon_image = Image.open("icons/playlist-repeat.png")
icon_image = icon_image.resize((30, 30))  # Resize the icon image to desired size
icon_photo_3 = ImageTk.PhotoImage(icon_image)

# Load the icon image - Results
icon_image = Image.open("icons/seo-search_1.png")
icon_image = icon_image.resize((30, 30))  # Resize the icon image to desired size
icon_photo_4 = ImageTk.PhotoImage(icon_image)

# Create button to browse and upload video
upload_button = ct.CTkButton(canvas, text="UPLOAD VIDEO ", width=230, height=60, command=open_file_dialog, font=('Roboto', 18), fg_color="#0652DD", bg_color="black",compound=tk.LEFT, image=icon_photo_1, hover_color="#0652DD")
upload_button.pack(padx=30, pady=30, anchor=tk.CENTER)

# Create button to browse and upload video
youtube_button = ct.CTkButton(canvas, text="YOUTUBE VIDEO", width=230, height=60, command=youtube_page, font=('Roboto', 18), fg_color="#FF0000", bg_color="black", compound=tk.LEFT, image=icon_photo, hover_color="#FF0000")
youtube_button.pack(padx=30, pady=30, anchor=tk.CENTER)

# Create button to restart video upload process
restart_button = ct.CTkButton(frame, text="RESTART", width=200, height=60, command=restart_upload, font=('Roboto', 18), fg_color="#10ac84", compound=tk.LEFT, image=icon_photo_3, bg_color="black", hover_color="#10ac84")
restart_button.pack(side=tk.LEFT,  padx=(0, 20)) # Hide the "Restart" button initially

# Initialize video capture object
cap = None

# Function to view results
def view_results():

    if cap is not None:
        video_path = file_path  # Get the video file path from the variable
        print("Video Path:", video_path)

    faces = face_extractor.process_video(os.path.abspath(file_path))
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


# Create button to view results
view_results_button = ct.CTkButton(frame, text="VIEW RESULTS", width=200, height=60, command=view_results, font=('Roboto', 18), compound=tk.LEFT, image=icon_photo_4, fg_color="#10ac84", bg_color="black", hover_color="#10ac84")
view_results_button.pack(side=tk.RIGHT,  padx=(20, 0))  # Hide the "View Results" button initially

# Create button to go back
back_button = ct.CTkButton(frame, text="", width=50, height=25, command=go_back, font=('Roboto', 16), fg_color="black", bg_color="black", hover_color="black", border_color="white", compound=tk.LEFT, image=icon_photo_2)
back_button.pack(side=tk.LEFT)

#root.resizable(False, False)
# Run the main loop
root.mainloop()