import os
import moviepy.editor as mp
def Frames_convert(image_dir,output_path,untamed):
    # Directory containing images
    image_paths = [os.path.join(image_dir, i) for i in os.listdir(image_dir) if i.endswith(('png', 'jpg', 'jpeg'))]
    video = mp.VideoFileClip(untamed)
    # Calculate duration per image
    original_duration = video.duration # seconds
    number_of_images = len(image_paths)
    duration_per_image = original_duration / number_of_images

    # Create ImageClips with the calculated duration
    images_clips = [mp.ImageClip(img).set_duration(duration_per_image) for img in image_paths]

    # Print the number of images and their total duration
    total_duration = sum(clip.duration for clip in images_clips)
    print(f'Number of Images: {number_of_images}')
    print(f'Total Duration: {total_duration} seconds')

    # Concatenate clips
    video = mp.concatenate_videoclips(images_clips, method='compose')

    # Write the final video to a file with the correct frame rate
    video.write_videofile(output_path, fps=25)
