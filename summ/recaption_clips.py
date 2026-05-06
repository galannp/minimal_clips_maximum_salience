from utils.movie_list import get_movie_list
from dl_utils.misc import check_dir
import os, json
from mllm.prompt_gemini_video import PrompterVL
from moviepy.editor import VideoFileClip
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_selection',type=str,choices=['random', 'silent', 'ours', 'upperbound'], default='ours')
    parser.add_argument('--nb_clips', type=int, default=50)
    parser.add_argument('--api_key',type=str)
    ARGS = parser.parse_args()

    include_transcripts = True # False
    captioning_version = 'gemini-2.5-flash-lite'
    if ARGS.clip_selection == 'upperbound':
        input_path = 'SummScreen/gt_video_clips_gemini-2.0-flash'
        clips_path = None
        output_path = f'gt_clips_{captioning_version}_captions'
    else:
        input_path = 'SummScreen/zero_shot_clips'
        if ARGS.clip_selection == 'random':
            clips_path = f'SummScreen/random_clips_nb_clips_{ARGS.nb_clips}'
        elif ARGS.clip_selection == 'silent':
            clips_path = f'SummScreen/silent_clips_nb_clips_{ARGS.nb_clips}'
        else:
            clips_path = f'SummScreen/video_agent_clips_id_{ARGS.nb_clips}_step_10_llm_gemini_qwen_omni_local'
        output_path = f'zero_shot_clips_{captioning_version}_captions'

    prompter = PrompterVL(version=captioning_version, key=ARGS.api_key)

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
            if clips_path is not None:
                for caption_tmstp in open(f'{clips_path}/{movie_name}').read().split('\n'):
                    caption_tmstp = int(caption_tmstp)
                    video_path = f'{input_path}/{movie_name}/clip_{caption_tmstp}.mp4'
                    if f'Caption {caption_tmstp}:' not in previous_captions:
                        if include_transcripts:
                            start, end = caption_tmstp - 10000, caption_tmstp + 10000
                            transcripts_clip = extract_transcripts_clip(movie_name, start, end)
                            prompt = 'Describe both the video, action and dialogue in one paragraph'
                            prompt = f'Transcripts:\n\n{transcripts_clip}\n\n{prompt}'
                        response = prompter.prompt(video_path, prompt)
                        print(response)
                        f.write(f'Caption {caption_tmstp}: {response}\n')
            else:
                for video_path in os.listdir(f'{input_path}/{movie_name}'):
                    video_path = f'{input_path}/{movie_name}/{video_path}'
                    caption_tmstp = int(video_path[:-4].split('_')[-1])
                    if f'Caption {caption_tmstp}:' not in previous_captions:
                        if include_transcripts:
                            vid = VideoFileClip(video_path)
                            duration_video = vid.duration * 1000
                            start, end = caption_tmstp - int(duration_video / 2), caption_tmstp + int(duration_video / 2)
                            transcripts_clip = extract_transcripts_clip(movie_name, start, end)
                            prompt = 'Describe both the video, action and dialogue in one paragraph'
                            prompt = f'Transcripts:\n\n{transcripts_clip}\n\n{prompt}'
                        response = prompter.prompt(video_path, prompt)
                        print(response)
                        f.write(f'Caption {caption_tmstp}: {response}\n')
