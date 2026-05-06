from moviepy.editor import VideoFileClip
import os
from dl_utils.misc import check_dir
import subprocess
from utils.movie_list import get_movie_list

def extract_clip(input_path, start_time, duration, output_path):
    command = [
        "/home/users/industry/cnrsatcreate/gpennec/ffmpeg-git-20240301-amd64-static/ffmpeg",
        "-ss", str(start_time),     # Start time
        "-i", input_path,           # Input file
        "-t", str(duration),        # Duration
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(command, check=True)

def split_video_into_clips(movie_list, clip_duration=20):
    for movie_name in movie_list:
        if not os.path.exists(f'SummScreen/zero_shot_clips/{movie_name}'):
            check_dir(f'SummScreen/zero_shot_clips/{movie_name}')
            # Load the video
            video_path = f'SummScreen/videos/{movie_name}.mp4'
            video = VideoFileClip(video_path)
            video_duration = int(video.duration)  # in seconds

            # List to keep track of clip file paths
            clip_paths = []

            # Generate 20-second clips
            for start in range(0, video_duration, clip_duration):
                end = min(start + clip_duration, video_duration)
                clip = video.subclip(start, end)
                clip_tmstp = int(1000 * (start + end) / 2)
                output_path = f"SummScreen/zero_shot_clips/{movie_name}/clip_{clip_tmstp}.mp4"
                extract_clip(video_path, start, clip_duration, output_path)

# Example usage
movie_list = get_movie_list(list_id=0)
clips = split_video_into_clips(movie_list)
print("Created clips:", clips)


# Improvements
# How to better decompose the whole video into clips (clip boundaries, scene dectection)