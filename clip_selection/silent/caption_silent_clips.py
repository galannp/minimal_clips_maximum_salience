from utils.movie_list import get_movie_list
from dl_utils.misc import check_dir
import os, json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--api_key',type=str)
parser.add_argument('--captioning_model',type=str,choices=['qwen', 'gemini'], default='qwen')
ARGS = parser.parse_args()

include_transcripts = False # True
input_path = 'SummScreen/selected_clips'

if ARGS.captioning_model == 'qwen':
    from mllm.prompt_qwen_omni import Qwen_Omni
    output_path = 'selected_clips_qwen_omni_captions'
    prompter = Qwen_Omni(params=7)
    prompt = 'Describe both the action and Summarize the corresponding dialogue.'
else:
    from mllm.prompt_gemini_video import PrompterVL
    captioning_version = 'gemini-2.5-flash-lite'
    output_path = f'selected_clips_{captioning_version}_captions'
    prompter = PrompterVL(version=captioning_version, key=ARGS.api_key)
    prompt = 'Describe both the video, action and dialogue in one paragraph'

def get_aligned_transcript(aligned_screenplay):
    return [utt for utt in aligned_screenplay if not utt[2].startswith('Caption')]

def extract_transcripts_clip(movie_name, start, end):
    with open(f'SummScreen/aligned_screenplays/{movie_name}.json') as f:
        aligned_transcripts = get_aligned_transcript(json.load(f)['Transcript'])
    transcripts_clip = []
    i = 0
    is_start = False
    while i < len(aligned_transcripts):
        utt = aligned_transcripts[i]
        if utt[0] >= start and utt[1] <= end:
            if not is_start:
                is_start = True
                if i > 0:
                    if ':' in aligned_transcripts[i - 1][2]:
                        transcripts_clip.append(aligned_transcripts[i - 1][2].split(':')[1])
                    else:
                        transcripts_clip.append(aligned_transcripts[i - 1][2])
            if ':' in utt[2]:
                transcripts_clip.append(utt[2].split(':')[1])
            else:
                transcripts_clip.append(utt[2])
        elif is_start:
            is_start = False
            if ':' in utt[2]:
                transcripts_clip.append(utt[2].split(':')[1])
            else:
                transcripts_clip.append(utt[2])
        i += 1
    return '\n'.join(transcripts_clip)

check_dir(output_path)
for movie_name in get_movie_list(list_id=0):
    print(movie_name)
    if movie_name not in open('active_threads_hierarchical.txt').read().split('\n'):
        with open('active_threads_hierarchical.txt', 'a') as f:
            f.write(f'{movie_name}\n')
        if os.path.exists(f'{output_path}/{movie_name}.txt'):
            previous_captions = open(f'{output_path}/{movie_name}.txt').read()
        else:
            previous_captions = ''
        f = open(f'{output_path}/{movie_name}.txt', 'a')
        for video_path in os.listdir(f'SummScreen/selected_clips/{movie_name}'):
            tmstp_0, tmstp_1 = video_path[:-4].split('_')[-2:]
            tmstp_0, tmstp_1 = int(tmstp_0), int(tmstp_1)
            caption_tmstp = int((tmstp_0 + tmstp_1) / 2)
            if f'Caption {caption_tmstp}:' not in previous_captions:
                if include_transcripts:
                    transcripts_clip = extract_transcripts_clip(movie_name, tmstp_0, tmstp_1)
                    prompt = f'Transcripts:\n\n{transcripts_clip}\n\n{prompt}'
                response = prompter.prompt(f'SummScreen/selected_clips/{movie_name}/{video_path}', prompt)
                print(response)
                f.write(f'Caption {caption_tmstp}: {response}\n')
