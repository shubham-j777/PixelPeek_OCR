from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess

# Create the Tkinter application window
root = Tk()
root.title("PixelPeek OCR Application")

# Set window size to a fixed value
root.geometry("800x600")

# Function to handle window resize event
def resize_bg(event):
    # Resize the background image to match the window size
    resized_bg_image = bg_image.resize((event.width, event.height), Image.LANCZOS)
    # Update the background image on the canvas
    canvas.img = ImageTk.PhotoImage(resized_bg_image)
    canvas.itemconfig(bg_image_item, image=canvas.img)
    # Update the canvas size to match the window size
    canvas.config(width=event.width, height=event.height)

    
def change_image_size(value):
    global detections_photo
    new_width = int(value)
    # Load the image and calculate the corresponding height to preserve aspect ratio
    detections_image = Image.open("outputs/detections.jpg")
    width, height = detections_image.size
    aspect_ratio = width / height
    new_height = int(new_width / aspect_ratio)
    # Resize the image using Lanczos filter for upscaling
    detections_image = detections_image.resize((new_width, new_height), Image.LANCZOS)
    detections_photo = ImageTk.PhotoImage(detections_image)
    detections_label.config(image=detections_photo)
    detections_label.image = detections_photo  # Keep a reference to prevent garbage collection


# Function to display loading indication
def show_loading():
    global loading_window
    loading_window = Toplevel(root)
    loading_window.title("Loading")
    loading_window.geometry("200x100")
    loading_window.transient(root)  # Make loading popup transient with respect to root window
    Label(loading_window, text="Loading...", font=("Arial", 14)).pack(pady=20)
    root.update()

# Function to close loading popup
def close_loading():
    loading_window.destroy()
    root.update()

# Function to display detection complete message
def show_detection_complete():
    messagebox.showinfo("Detection Complete", "Detection process is complete!")

# Load the background image
bg_image = Image.open("bg_image.jpg")

# Create a Canvas widget
canvas = Canvas(root)
canvas.pack(fill=BOTH, expand=YES)

# Add the background image to the canvas
bg_image_item = canvas.create_image(0, 0, anchor=NW, image=None)

# Bind the resize event to the function
canvas.bind("<Configure>", resize_bg)

# Function to open file dialog and select image
def only_OCR_paddleOCR():
    show_loading()  # Show loading indication
    subprocess.run(['python', 'paddle_ocr.py'])
    close_loading()  # Close loading indication
    show_detection_complete()  # Show detection complete message

def only_OCR_pytesseractOCR():
    show_loading()  # Show loading indication
    subprocess.run(['python', 'pytesseract_ocr.py'])
    close_loading()  # Close loading indication
    show_detection_complete()  # Show detection complete message

def ocr_with_table(): 
    show_loading()  # Show loading indication
    subprocess.run(['python', 'ocr_table.py'])
    close_loading()  # Close loading indication
    show_detection_complete()  # Show detection complete message

def show_text(): 
    file_path = r"outputs\detected_texts.txt"
    if file_path:
        with open(file_path, "r") as file:
            text = file.read()
            text_win.delete("1.0", END)  
            text_win.insert(END, text)  

    # Load the image and display it in the image widget
    detections_image = Image.open("outputs/detections.jpg")
    detections_photo = ImageTk.PhotoImage(detections_image)
    detections_label.config(image=detections_photo)
    detections_label.image = detections_photo  # Keep a reference to prevent garbage collection
    

# Create a title banner spanning the entire width
title_banner = Label(root, text="PixelPeek OCR Application", font=("Hi Jack", 30), bg="#1a747d", fg="#e2e2e2", pady=10)
title_banner.place(relx=0.5, rely=0.1, anchor=CENTER, relwidth=1.0)

btn_open_image = Button(root, text="Only OCR (paddle OCR)", command=only_OCR_paddleOCR, width=25, height=2,bg="#D1E8E2") 
btn_open_image.place(relx=0.2, rely=0.2, anchor=CENTER)

btn_open_camera = Button(root, text="Only OCR (pytesseract OCR)", command=only_OCR_pytesseractOCR, width=25, height=2,bg="#CAF0F8") 
btn_open_camera.place(relx=0.5, rely=0.2, anchor=CENTER)

btn_detect_text = Button(root, text="OCR with Table", command=ocr_with_table, width=25, height=2,bg="#A9D6E5")
btn_detect_text.place(relx=0.8, rely=0.2, anchor=CENTER)

# Button to detect text
btn_detect_text = Button(root, text="Detect Text", command=show_text, width=25, height=2,bg="#90E0EF"   )
btn_detect_text.place(relx=0.2, rely=0.3, anchor=CENTER)

# Slider to change the image size
size_slider = Scale(root, from_=50, to=2000, orient=HORIZONTAL, command=change_image_size, length=200)
size_slider.place(relx=0.55, rely=0.3, anchor=CENTER)

text_win = Text(root, width=40, height=15, bg="#D4D4D4")
text_win.place(relx=0.01, rely=0.4, anchor=NW)

# Label to display the detections image
detections_label = Label(root)
detections_label.place(relx=0.99, rely=0.4, anchor=NE)

# Run the Tkinter event loop
root.mainloop()
