import argparse
import os
import shutil
from dl_utils.misc import check_dir
import json
from glob import glob
import numpy as np
import pickle
import subprocess
from moviepy.editor import VideoFileClip
from utils.movie_list import get_movie_list

def get_video_duration(file_path):
    try:
        # Load the video file
        video = VideoFileClip(file_path)
        
        # Get the duration in seconds
        duration = video.duration
        
        # Close the video file to release resources
        video.close()

        return duration
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def extract_clip(input_file, start_time, duration, output_file):
    """
    Extracts a clip from a video using ffmpeg.

    Parameters:
        input_file (str): Path to the input video file.
        start_time (str): Start time of the clip in format 'hh:mm:ss'.
        duration (str): Duration of the clip in format 'hh:mm:ss'.
        output_file (str): Path to save the extracted clip.
    """
    try:
        command = [
            'ffmpeg',
            '-i', input_file,
            '-ss', start_time,  # Seek to the start time
            '-t', duration,     # Clip duration
            output_file
        ]

        # Run the ffmpeg command
        subprocess.run(command, check=True)
        print(f"Clip successfully extracted to {output_file}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while extracting the clip: {e}")

def get_aligned_transcript(aligned_screenplay):
    return [utt for utt in aligned_screenplay if not utt[2].startswith('Caption')]

def select_frames_to_caption(epname, extract):
    with open(f'{ARGS.path_aligned}/{epname}.json') as f:
        aligned_screenplay = json.load(f)['Transcript']
    aligned_transcripts = get_aligned_transcript(aligned_screenplay)

    only_utterances = aligned_transcripts
    #only_utterances = [line for line in aligned_transcripts if ':' in line[2] and not line[2].startswith('[')]

    # Define the input and output file paths
    input_video_path = f'SummScreen/videos/{epname}.mp4'  # Replace with your video file path
    duration_video = get_video_duration(input_video_path)

    # Load the video file
    if ARGS.max_tot_dur_clips is not None:
        folder = f'selected_clips_max_dur_{ARGS.max_tot_dur_clips}_exact'
    elif ARGS.nb_clips is not None:
        folder = f'selected_clips_nb_clips_{ARGS.nb_clips}_exact'
    elif ARGS.gap_tmstp_threshold != 15:
        folder = f'selected_clips_{ARGS.gap_tmstp_threshold}_exact'
    else:
        folder = f'selected_clips_exact'
    if ARGS.path_aligned != 'SummScreen/aligned_screenplays':
        folder += f'_{ARGS.path_aligned.split("/")[-1]}'

    out_tmstp_clips = f'SummScreen/{folder}_tmstp'
    check_dir(out_tmstp_clips)

    tmstp_clips = []
    previous_durations = []
    for i in range(len(only_utterances) - 1):
        clip_start, clip_end = only_utterances[i][1], only_utterances[i + 1][0]
        if ARGS.nb_clips is not None or clip_end - clip_start > ARGS.gap_tmstp_threshold * 1000:
            # A clip is at least 20 seconds long
            previous_duration = clip_end - clip_start
            previous_durations.append(previous_duration)
            #added_time = (20000 - previous_duration) / 2 #max((20000 - previous_duration) / 2, 0)
            #clip_start, clip_end = max(clip_start - added_time, 0), min(clip_end + added_time, duration_video * 1000)
            '''if random:
                dur = clip_end - clip_start
                stt = np.random.randint(int(duration_video * 1000) - dur + 1)
                tmstp_clips.append((stt, stt + dur))
            else:'''
            tmstp_clips.append((clip_start, clip_end))

    if ARGS.nb_clips is not None or ARGS.max_tot_dur_clips is not None:
        _, tmstp_clips = map(list, zip(*sorted(list(zip(previous_durations, tmstp_clips)), reverse=True)))
    if ARGS.nb_clips is not None:
        tmstp_clips = sorted(tmstp_clips[:ARGS.nb_clips])
    if ARGS.max_tot_dur_clips is not None:
        durations_clips = [a[1] - a[0] for a in tmstp_clips]
        cap = 0
        count_dur = 0
        for d in durations_clips:
            count_dur += d
            cap += 1
            if count_dur > ARGS.max_tot_dur_clips:
                break
        print(f"Cap at {count_dur} millisec")
        tmstp_clips = sorted(tmstp_clips[:cap])

    np.save(f'{out_tmstp_clips}/{epname}.npy', tmstp_clips)

    nb_clips = len(tmstp_clips)
    print(f'Episode {epname}: {nb_clips} selected clips for captionning')
    print(f'ratio of captions in the screenplay {nb_clips / (len(only_utterances) - 1)}')

    if extract:
        for idx, (start_time, end_time) in enumerate(tmstp_clips):
            start_time_sec, end_time_sec = start_time / 1000, end_time / 1000
            duration = end_time_sec - start_time_sec

            # Define the output file path and VideoWriter
            check_dir(f'SummScreen/{folder}/{epname}')
            output_file = f"SummScreen/{folder}/{epname}/clip_{int((start_time + end_time) / 2)}.mp4"
            # Extract the subclip
            print(f"Extracting clip {idx + 1}: {start_time_sec}s to {end_time_sec}s...")
            extract_clip(input_video_path, str(start_time_sec), str(duration), output_file)

    '''# Load the video file

    for idx, (start_time, end_time) in enumerate(tmstp_clips):
        video = VideoFileClip(input_video_path)
        start_time_sec, end_time_sec = start_time / 1000, end_time / 1000

        # Define the output file path and VideoWriter
        check_dir(f'SummScreen/selected_clips/{epname}')
        output_file = f"SummScreen/selected_clips/{epname}/clip_{int((start_time + end_time) / 2)}.mp4"
        # Extract the subclip
        clip = video.subclip(start_time_sec, end_time_sec)

        print(f"Extracting clip {idx + 1}: {start_time_sec}s to {end_time_sec}s...")

        # Write the subclip to the output file
        clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

        # Close the clips to free resources
        clip.close()
        print(f"Clip {idx + 1} saved as {output_file}")

        # Release the video capture object
        video.close()
        print("All clips have been successfully extracted.")'''

def get_sorted_keyframes(root_dir):
    keyframes_files = glob('**/*.jpg', root_dir=root_dir, recursive=True)
    timestamps = [int(fn.split('/')[-1][:-4]) for fn in keyframes_files]
    sorted_keyframes_and_timestamps = sorted(zip(timestamps, keyframes_files))
    timestamps, keyframes = list(map(list, zip(*sorted_keyframes_and_timestamps)))
    return timestamps, keyframes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap_tmstp_threshold', type=int, default=0)
    parser.add_argument('--nb_clips', type=int, default=None)
    parser.add_argument('--max_tot_dur_clips', type=int, default=None)
    parser.add_argument('--epname', type=str,nargs='+', default=['movie_list'], choices=['all', 'movie_list'])
    parser.add_argument('--extract',default=False)
    parser.add_argument('--path_aligned',default='SummScreen/aligned_screenplays')
    ARGS = parser.parse_args()

    if ARGS.max_tot_dur_clips is not None:
        folder = f'selected_clips_max_dur_{ARGS.max_tot_dur_clips}_exact_tmstp'
    elif ARGS.nb_clips is not None:
        folder = f'selected_clips_nb_clips_{ARGS.nb_clips}_exact_tmstp'
    elif ARGS.gap_tmstp_threshold != 15:
        folder = f'selected_clips_{ARGS.gap_tmstp_threshold}_exact_tmstp'
    else:
        folder = f'selected_clips_exact_tmstp'
    if ARGS.path_aligned != 'SummScreen/aligned_screenplays':
        folder += f'_{ARGS.path_aligned.split("/")[-1]}'

    if ARGS.epname == ['all']:
        ARGS.epname = [en[:-4] for en in os.listdir('SummScreen/videos')]
    elif ARGS.epname == ['movie_list']:
        ARGS.epname = get_movie_list(list_id=0)
    for en in ARGS.epname:
        print(f'SummScreen/{folder}/{en}.npy')
        if not os.path.exists(f'SummScreen/{folder}/{en}.npy'):
            print(f'Selecting keyframes to caption for {en}')
            select_frames_to_caption(en, extract=ARGS.extract)
        else:
            print(f'Selected keyframes already exist for {en}')
