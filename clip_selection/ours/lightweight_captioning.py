# conda activate vllm
import cv2
import PIL.Image
import os
import json
import shutil
import math
from openai import OpenAI
import base64
from dl_utils.misc import check_dir
from utils.movie_list import get_movie_list
from time import sleep
#from smolvlm2_prompt import SPrompter

def encode_video(video_path):
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")

def extract_frames(video_path, frame_interval=5):
    '''
    Extracts frames from a video at the specified interval.

    Args:
        video_path (str): Path to the input video file.
        interval (int): Interval in seconds between extracted frames.

    Returns:
        None
    '''
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    #if not cap.isOpened():
    #    print("Error: Cannot open video.")
    #    return

    # Get the total number of frames in the video
    fps = round(cap.get(cv2.CAP_PROP_FPS))

    # Ensure the output folder exists
    frame_count = 0
    saved_count = 0

    all_frames_pth = []
    frames_tmp_dir = 'frames_tmp'
    if os.path.exists(frames_tmp_dir):
        shutil.rmtree(frames_tmp_dir)
        os.mkdir(frames_tmp_dir)
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Exit if no more frames

        if frame_count % (fps * frame_interval) == 0:
            new_frame_fn = f'{frames_tmp_dir}/{frame_count}.png'
            cv2.imwrite(new_frame_fn, frame)
            all_frames_pth.append(new_frame_fn)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames.")

    return all_frames_pth

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

def try_predict(client, **kwargs):
    got_response = False
    while not got_response:
        try:
            result = client.predict(**kwargs)
            got_response = True
        except Exception as e:
            print(e)
            print('Retry requesting API!!!')
    return result

# Example usage
#analyze = 'audio'
analyze = 'video'
model = 'qwen_omni'
local = True #False
if local:
    from mllm.prompt_qwen_omni import Qwen_Omni
    params = 3 #params = 7
    qwen_omni = Qwen_Omni(params=params)
#model = 'qwen'
#model = 'smolvlm2'
include_transcripts = False # True
retry_generate = False
continue_generate = False
zero_shot_clips = 'zero_shot_clips' # 'zero_shot_scenes'

#api_key = 'sk-6e4d1ff20d274799a5eb46503c175f94' # Jianyu
#api_key = 'sk-3d4b4ff506334118bedcdb937db2dc91' # Galann JCLT
#api_key = 'sk-2c7b21a0696944209b4d89b37f1b1f48' # Katayoun
#api_key = 'sk-c065c5a155b04c6dac69e5f9f843cf7d' # Matthias
#api_key = 'sk-47f1e8c4df464c0ab8f31de792a1ffe2' # Yan
#api_key = 'sk-aee394fb0c5b4a219b8d6d78cf92dcec' # Arjun
#api_key = 'sk-21aa2fd6d6ec42da9365ea321c94d48f' # Galann CNRS
#api_key = 'sk-79237f705b1149a1a485923c79737895' # Galann gmail
#client = OpenAI(api_key=api_key, base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

for movie_name in get_movie_list(list_id=0):
    if movie_name not in open('active_threads_hierarchical.txt').read().split('\n'):
        with open('active_threads_hierarchical.txt', 'a') as f:
            f.write(f'{movie_name}\n')
        if analyze == 'audio':
            zero_shot_clips = 'zero_shot_clips_audio'
        path = f'SummScreen/{zero_shot_clips}/{movie_name}'
        if os.path.exists(path):
            out_path = f'{zero_shot_clips}_{model}_captions'
            if include_transcripts:
                out_path += '_with_tr'
            if local and params != 7:
                out_path += f'_params_{params}'
            in_path = out_path + f'/{movie_name}.txt'
            if os.path.exists(in_path):
                old_generated_captions = open(out_path + f'/{movie_name}.txt').read()
            else:
                old_generated_captions = ''
            if retry_generate:
                out_path += '_retry_generate'
            check_dir(out_path)
            out_path += f'/{movie_name}.txt'
            print(movie_name)
            if model == 'smolvlm2':
                videos = sorted(os.listdir(path), key=lambda video_file: int(video_file[:-4].split('_')[1]))
                captions_id = [int(video_file[:-4].split('_')[1]) for video_file in videos]
                #prompt = 'Describe both the action and Summarize the corresponding dialogue.'
                prompt = 'Summarize the provided Transcripts.'
                transcripts = [extract_transcripts_clip(movie_name, caption_id - 10000, caption_id + 10000) for caption_id in captions_id]
                prompts = [f'Transcripts:\n\n{transcript_clip}\n\n{prompt}' for transcript_clip in transcripts]
                os.makedirs('SummScreen/zero_shot_clip_captions_smol/', exist_ok=True)
                videos = [f'{path}/{v}' for v in videos]
                open(f'SummScreen/zero_shot_clip_captions_smol/{movie_name}.txt', 'w').write('\n'.join(SPrompter().prompt(videos, prompts)))
            elif model.startswith('qwen'):
                if old_generated_captions != '':
                    for line in old_generated_captions.split('\n')[::-1]:
                        if line.startswith('Caption'):
                            last_caption_id = line.split(':', maxsplit=1)[0].split('Caption')[1].strip()
                            break
                else:
                    last_caption_id = None
                for video_file in sorted(os.listdir(path), key=lambda video_file: int(video_file[:-4].split('_')[1])):
                    caption_id = video_file[:-4].split('_', maxsplit=1)[1]
                    if (video_file[-4:] == '.mp3' or video_file[-4:] == '.mp4') and (last_caption_id is None or ((continue_generate and not retry_generate and int(caption_id.split('_')[1]) > int(last_caption_id.split('_')[1])) or retry_generate and f'Caption {caption_id}:' not in old_generated_captions)):
                        print(f'Generating for {video_file}')
                        video_path = f'{path}/{video_file}'

                        ii = 0
                        got_response = False
                        while not got_response and ii < 1:
                            #try:
                            if analyze == 'audio':
                                prompt = 'What sounds do you hear? Is there something worth noting? Does it suggest something important is happening?'
                                
                                base64_audio = encode_audio(video_path)
                                completion = client.chat.completions.create(
                                    model="qwen2.5-omni-7b",
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "input_audio",
                                                    "input_audio": {
                                                        "data": f"data:;base64,{base64_audio}",
                                                        "format": "mp3",
                                                    },
                                                },
                                                {"type": "text", "text": prompt},
                                            ],
                                        },
                                    ],
                                    # Set output data modalities, currently supports two types: ["text","audio"], ["text"]
                                    modalities=["text"],
                                    # stream must be set to True, otherwise an error will occur
                                    stream=True,
                                    stream_options={"include_usage": True},
                                )
                                response = ''
                                for chunk in completion:
                                    if chunk.choices:
                                        content = chunk.choices[0].delta.content
                                        if content is not None:
                                            response += content
                                print(response)
                            elif analyze == 'video' and model == 'qwen_omni':
                                #prompt = 'What sounds do you hear? Is there something worth noting? Does it suggest something important is happening?'
                                #prompt = 'Describe the action in a few sentences. Is there something worth noting? Does it suggest something important is happening?'

                                if not local:
                                    prompt = 'Describe both the action and Summarize the corresponding dialogue in a few sentences.'
                                else:
                                    prompt = 'Describe both the action and Summarize the corresponding dialogue.'
                                #prompt = 'Describe the action in a few sentences. When you describe the action, always identify the name of the characters whenever possible.'
                                if include_transcripts:
                                    start, end = caption_id - 10000, caption_id + 10000
                                    transcripts_clip = extract_transcripts_clip(movie_name, start, end)
                                    prompt = f'Transcripts:\n\n{transcripts_clip}\n\n{prompt}'

                                if local:
                                    response = qwen_omni.prompt(video_path, prompt)
                                    print(response)
                                else:
                                    base64_video = encode_video(video_path)
                                    completion = client.chat.completions.create(
                                        model="qwen2.5-omni-7b",
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": [
                                                    {
                                                        "type": "video_url",
                                                        "video_url": {
                                                            "url": f"data:;base64,{base64_video}",
                                                        },
                                                    },
                                                    {"type": "text", "text": prompt},
                                                ],
                                            },
                                        ],
                                        # Set output data modalities, currently supports two types: ["text","audio"], ["text"]
                                        modalities=["text"],
                                        # stream must be set to True, otherwise an error will occur
                                        stream=True,
                                        stream_options={"include_usage": True},
                                    )

                                    response = ''
                                    for chunk in completion:
                                        #print(chunk.usage)
                                        if chunk.choices:
                                            content = chunk.choices[0].delta.content
                                            if content is not None:
                                                response += content
                                    print(response)

                            elif analyze == 'video' and model == 'qwen':
                                sample_files = extract_frames(video_path)
                                sample_files = [encode_image(sp) for sp in sample_files]

                                prompt = 'Describe both the action and Summarize the corresponding dialogue in a few sentences.'
                                if include_transcripts:
                                    start, end = caption_id - 10000, caption_id + 10000
                                    transcripts_clip = extract_transcripts_clip(movie_name, start, end)
                                    prompt = f'Transcripts:\n\n{transcripts_clip}\n\n{prompt}'

                                completion = client.chat.completions.create(
                                    model="qwen2.5-vl-3b-instruct",
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/jpeg;base64,{sp}"
                                                    },
                                                } for sp in sample_files] + [{"type": "text", "text": prompt}],
                                        }
                                    ],
                                )

                                response = completion.choices[0].message.content
                                print(response)

                            with open(out_path, 'a') as f:
                                f.write(f'Caption {caption_id}: {response}\n')
                            got_response = True

                            #except Exception as e:
                            #    print(e)
                            #    ii += 1
                            #    if 'inappropriate' in str(e):
                            #        pass
                            #    else:
                            #        sleep(20)
                            #        pass
