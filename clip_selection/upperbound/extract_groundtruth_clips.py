import os
import json
from dl_utils.misc import check_dir
from moviepy.editor import VideoFileClip, concatenate_videoclips
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

if __name__ == '__main__':
    movie_list = get_movie_list(list_id=0)
    merge_adjacent = False
    version = 'gemini-2.0-flash'
    out_path = f'SummScreen/gt_video_clips_{version}'
    check_dir(out_path)
    avg_dur = 0
    i = 0
    for video_file in os.listdir('SummScreen/videos'):
        if video_file[:-4] in movie_list and video_file[:-4] not in os.listdir(out_path) and video_file[:-4] in os.listdir(f'groundtruth_clips_{version}'):
            # Load the full video
            video = VideoFileClip(f'SummScreen/videos/{video_file}')
            duration_video = video.duration * 1000
            check_dir(f'{out_path}/{video_file[:-4]}')
            if video_file[:-4] not in os.listdir(f'{out_path}/{video_file[:-4]}'):
                with open(f'groundtruth_clips_{version}/{video_file[:-4]}') as f:
                    gt_clips_id = list(map(int, f.read().split('\n')))
                with open(f'SummScreen/aligned_screenplays/{video_file[:-4]}.json') as f:
                    aligned_screenplay = json.load(f)['Transcript']
                all_start_ends = []
                for utt in aligned_screenplay:
                    if utt[2].startswith('Caption'):
                        caption_id = int(utt[2].split(':', maxsplit=1)[0].split('Caption', maxsplit=1)[1].strip())
                        if caption_id in gt_clips_id:
                            #start, end = max(0, utt[0] - 10000) / 1000, min(utt[1] + 10000, duration_video) / 1000
                            start, end = utt[0] / 1000, utt[1] / 1000
                            if start * 1000 > duration_video:
                                print('Warning: clip goes beyond the max duration of the video!!!!!')
                                continue
                            if len(all_start_ends) > 0:
                                if start == all_start_ends[-1][0]:
                                    if end > all_start_ends[-1][1]:
                                        all_start_ends[-1][1] = end
                                    else:
                                        continue
                                elif merge_adjacent and start <= all_start_ends[-1][1]:
                                    all_start_ends[-1][1] = end
                                else:
                                    all_start_ends.append([start, end])
                            else:
                                all_start_ends.append([start, end])

                i += 1
                for start, end in all_start_ends:
                    duration = end - start
                    avg_dur += duration
                    extract_clip(f'SummScreen/videos/{video_file}', start, duration, f"{out_path}/{video_file[:-4]}/clip_{int((start + end) / 2 * 1000)}.mp4")

    print(avg_dur / i)
