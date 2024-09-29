import os
import moviepy.editor as mp
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best_Yolov8x.pt')

# Create a Tkinter window
root = tk.Tk()
root.title("Video Frame Bounding Box")

# Define global variables
cap = None
frame_count = 0
canvas = None
photo = None
playing = False
paused = False
bbox_list = []  # To store bounding boxes and their info
selected_class = None  # Stores the currently selected class (Player/Ball)
current_frame = None  # Store the current frame when paused
save_processed_video = False  # Flag to determine if the video should be saved
output_video_name = 'test12-bbox.mp4'  # Default output video name
processed_frames = []  # List to store processed frames

def load_video():
    global cap, frame_count, playing, paused, bbox_list, current_frame, save_processed_video, output_video_name, processed_frames

    save_video = messagebox.askyesno("Save Video", "Do you want to save the processed video?")
    if save_video:
        save_processed_video = True
        
        # Ask the user to enter the video name
        output_video_name = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")], initialfile='test12-bbox.mp4')
        if not output_video_name:
            messagebox.showerror("Error", "Output video name must be provided.")
            return
    else:
        save_processed_video = False

    # Ask the user to select a video file
    video_path = filedialog.askopenfilename(title="Select a Video", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    
    if video_path:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        playing = True
        paused = False
        bbox_list = []  # Reset bbox list for the new video
        current_frame = None  # Reset current frame
        processed_frames = []  # Reset processed frames
        process_video(video_path)

def process_video(video_path):
    """Process video using YOLO model, store frames, and update Tkinter canvas."""
    global cap, frame_count, canvas, photo, playing, paused, bbox_list, selected_class, current_frame, processed_frames
    
    confidence_threshold = 0.10  # Set confidence threshold

    while cap.isOpened():
        if playing and not paused:
            ret, frame = cap.read()
            if not ret:
                break

            bbox_list = []  # Reset bbox_list for each frame

            # Run YOLO detection on the frame
            results = model(frame)

            # Loop over each detection
            for result in results:
                for box in result.boxes:
                    confidence = box.conf.cpu().numpy()[0]  # Extract confidence score

                    if confidence >= confidence_threshold:
                        # Extract bounding box coordinates (convert to int)
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                        # Extract class index and map it to class name
                        class_idx = int(box.cls.cpu().numpy())
                        class_name = model.names[class_idx]

                        # Store the bounding box
                        bbox_list.append((x1, y1, x2, y2, class_name, confidence))

                        # Draw bounding box and class label only for the selected class
                        if selected_class is None or class_name.lower() == selected_class.lower():
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{class_name} {confidence:.2f}", 
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            current_frame = frame.copy()  # Save current frame when paused

            # Add the processed frame to the list for video creation later
            processed_frames.append(frame)

            # Display current frame in Tkinter canvas
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(img)

            if canvas is None:
                canvas = tk.Canvas(root, width=img_tk.width(), height=img_tk.height())
                canvas.pack(side=tk.RIGHT)  # Place canvas on the right side
            else:
                canvas.delete("all")  # Clear canvas for new frame

            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            photo = img_tk  # Keep reference to prevent garbage collection

            frame_count += 1

        root.update()
        if paused:
            root.after(100)  # Wait while paused

    cap.release()

    if save_processed_video:
        # Convert frames stored in memory to a video using MoviePy
        convert_frames_to_video(processed_frames, output_video_name)
        
        # Ask if user wants to play the video with bounding boxes
        play_video = messagebox.askyesno("Play Video", "Do you want to play the video with bounding boxes?")
        if play_video:
            play_bounding_box_video(output_video_name)

def convert_frames_to_video(frames, output_path):
    """Convert a list of numpy frames to a video using moviepy."""
    clips = [mp.ImageClip(frame).set_duration(1/25) for frame in frames]  # Set frame duration for 25 fps
    video = mp.concatenate_videoclips(clips, method='compose')
    video.write_videofile(output_path, fps=25)

def play_bounding_box_video(video_path):
    """Function to play the video using OpenCV."""
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow('Bounding Box Video', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
            break
    video.release()
    cv2.destroyAllWindows()

def toggle_play_pause():
    global playing, paused
    if playing:
        playing = False
        paused = True
        play_pause_button.config(text="Play")
    else:
        playing = True
        paused = False
        play_pause_button.config(text="Pause")
        process_video('')  # Continue video processing

def select_class(class_name):
    """Sets the currently selected class."""
    global selected_class
    selected_class = class_name.lower()
    print(f"Selected class: {selected_class}")

# Set up the UI elements
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10)

load_button = tk.Button(left_frame, text="Load Video", command=load_video)
load_button.pack(pady=10)

play_pause_button = tk.Button(left_frame, text="Pause", command=toggle_play_pause)
play_pause_button.pack(pady=10)

# Selection pane for "Player" and "Ball"
player_button = tk.Button(left_frame, text="Player", command=lambda: select_class('Player'))
player_button.pack(pady=5)

ball_button = tk.Button(left_frame, text="Ball", command=lambda: select_class('Ball'))
ball_button.pack(pady=5)

# Reset button to show all classes
reset_button = tk.Button(left_frame, text="Reset", command=lambda: select_class(None))
reset_button.pack(pady=5)

# Start the Tkinter main loop
root.mainloop()
