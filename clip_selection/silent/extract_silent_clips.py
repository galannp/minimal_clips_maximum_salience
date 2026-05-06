from utils.movie_list import get_movie_list
import numpy as np

from moviepy.editor import VideoFileClip
import os
from dl_utils.misc import check_dir
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nb_clips', type=int, default=50)
ARGS = parser.parse_args()

selected_clips_path = f'SummScreen/selected_clips_nb_clips_{ARGS.nb_clips}_exact_tmstp'

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

for movie_name in get_movie_list(list_id=0):
    print(movie_name)
    if movie_name not in open('active.txt').read().split('\n'):
        open('active.txt', 'a').write(movie_name + '\n')
        all_tmstp = np.load(f'{selected_clips_path}/{movie_name}.npy')

        check_dir(f'SummScreen/selected_clips/{movie_name}')
        # Load the video
        video_path = f'SummScreen/videos/{movie_name}.mp4'
        video = VideoFileClip(video_path)
        video_duration = int(video.duration) # in seconds

        for start, end in all_tmstp:
            start_sec, end_sec = start / 1000, end / 1000
            if start_sec == end_sec:
                continue
            clip = video.subclip(start_sec, end_sec)
            output_path = f"SummScreen/selected_clips/{movie_name}/clip_{start}_{end}.mp4"
            if not os.path.exists(output_path):
                extract_clip(video_path, start_sec, end_sec - start_sec, output_path)
