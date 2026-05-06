# conda activate vllm

import cv2
from PIL import Image
import os
import json
import shutil
import time
from dl_utils.misc import check_dir
import cv2
import numpy as np
from utils.movie_list import get_movie_list
from google import genai
from google.genai import types
import argparse
from openai import OpenAI

def extract_frames(video_path, frame_interval=5):
    """
    Extracts frames from a video file.

    Args:
        video_path (str): Path to the video file.
        frame_interval (int): Extracts every Nth frame. Use 1 for every frame.
                            Increase this to sample frames and reduce input size.

    Returns:
        list: A list of PIL Image objects representing the extracted frames.
            Returns None if video cannot be opened.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    fps = round(cap.get(cv2.CAP_PROP_FPS))

    frame_count = 0
    while True:
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            break # End of video

        if frame_count % (fps * frame_interval) == 0:
            # Convert OpenCV BGR frame to RGB PIL Image
            # The API often works best with standard image formats/libraries
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)

        frame_count += 1

    cap.release()
    print(f"Extracted {len(frames)} frames (sampled every {frame_interval} frames).")
    return frames

def get_aligned_transcript(aligned_screenplay):
    return [utt for utt in aligned_screenplay if not utt[2].startswith('Caption')]  

def build_interleaved_context(movie_name, clips_dir, tmstp_clips, source, screenplay_path):
    if clips_dir is None:
        if source == 'screenplay':
            if 'utt' in screenplay_path or 'transcripts' in screenplay_path:
                key = 'Transcript'
            else:
                key = 'Screenplay'
            with open(f'{screenplay_path}/{movie_name}.json') as f:
                return ['\n'.join(json.load(f)[key])]
        else:
            with open(f'SummScreen/transcripts/{movie_name}.json') as f:
                return ['\n'.join(json.load(f)['Transcript'])]

    elif not tmstp_clips:
        with open(f'SummScreen/screenplays/{movie_name}.json') as f:
            screenplay = json.load(f)['Screenplay']

        gt_video_clips_path = f'{clips_dir}/{movie_name}'
        interleaved_context = []
        current_context = []
        for utt in screenplay:
            if utt.startswith('Caption'):
                caption_id = int(utt.split(':', maxsplit=1)[0].split('Caption')[1].strip())
                if current_context != []:
                    interleaved_context.append('\n'.join(current_context))
                    current_context = []
                if f'clip_{caption_id}.mp4' in os.listdir(gt_video_clips_path):
                    interleaved_context.extend(extract_frames(f'{gt_video_clips_path}/clip_{caption_id}.mp4'))
            else:
                current_context.append(utt)
        if current_context != []:
            interleaved_context.append('\n'.join(current_context))

        return interleaved_context

    else:
        with open(f'SummScreen/aligned_screenplays/{movie_name}.json') as f:
            aligned_screenplay = json.load(f)['Transcript']
        aligned_transcripts = get_aligned_transcript(aligned_screenplay)

        interleaved_context = []
        current_context = []
        i = 0
        random_video_clips_path = f'{clips_dir}/{movie_name}'
        random_clips = os.listdir(random_video_clips_path)
        tmstp_clips = [int(clip[:-4].split('_')[-1]) for clip in random_clips]
        for clip_tmstp, clip in sorted(list(zip(tmstp_clips, random_clips))):
            while i < len(aligned_transcripts) and aligned_transcripts[i][0] < clip_tmstp:
                current_context.append(aligned_transcripts[i][2])
                i += 1
            if current_context != []:
                interleaved_context.append('\n'.join(current_context))
                current_context = []
            interleaved_context.extend(extract_frames(f'{random_video_clips_path}/{clip}'))

        while i < len(aligned_transcripts):
            current_context.append(aligned_transcripts[i][2])
            i += 1
        if current_context != []:
            interleaved_context.append('\n'.join(current_context))

        return interleaved_context

class Qwen_next:
    def __init__(self):
        #api_key = 'sk-6e4d1ff20d274799a5eb46503c175f94' # Jianyu
        #api_key = 'sk-3d4b4ff506334118bedcdb937db2dc91' # Galann JCLT
        #api_key = 'sk-2c7b21a0696944209b4d89b37f1b1f48' # Katayoun
        #api_key = 'sk-c065c5a155b04c6dac69e5f9f843cf7d' # Matthias
        api_key = 'sk-47f1e8c4df464c0ab8f31de792a1ffe2' # Yan
        #api_key = 'sk-aee394fb0c5b4a219b8d6d78cf92dcec' # Arjun
        #api_key = 'sk-21aa2fd6d6ec42da9365ea321c94d48f' # Galann CNRS
        #api_key = 'sk-79237f705b1149a1a485923c79737895' # Galann gmail
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

    def prompt(self, contents):
        completion = self.client.chat.completions.create(
            model="qwen2.5-72b-instruct",#"qwen3-next-80b-a3b-instruct",
            messages=[
                {"role":"user","content":'\n\n'.join(contents)}
            ],
            stream=False,
            
        )
        return completion.choices[0].message.content


class PrompterVL():
    def __init__(self, version="gemini-1.5-pro", key='katayoun'):
        print('Initializing Gemini')
        if key == 'galann':
            # Galann's key
            self.client = genai.Client(api_key="AIzaSyCqrTgqpAPUfeRlKULCJ0_OSRcKWmWHl7k")
        elif key == 'yan':
            # Yan's key
            self.client = genai.Client(api_key="AIzaSyC2f8JKCIxWIR_ONd1XhnsnL_MrwfxMXh8")
        elif key == 'katayoun':
            # Katayoun's key
            self.client = genai.Client(api_key="AIzaSyAQhrS4eAnmhk-_cgc75IIt7klL0ePjq9M")
        elif key == 'matthias':
            # Matthias key
            self.client = genai.Client(api_key="AIzaSyC_JRRI8i2B_CIqaRSkjxJIvWMQiDQWFFE")
        elif key == 'ludo':
            self.client = genai.Client(api_key="AIzaSyA_75PUtR8ueKIr1cmQXdsUeBUbM8HFBtc")
        elif key == 'srecko':
            self.client = genai.Client(api_key="AIzaSyARLeIIZoBh4VF3eu4rNucZZ9zkpQ9BTiM")
        elif key == 'zhengyuan':
            self.client = genai.Client(api_key="AIzaSyDWc__lEuJom3izuOnX5eTVl-K3USvXMml")
        else:
            raise ValueError('Unrecognized Gemini API key')
        self.version = version

        # Set safety settings to allow all types of content
        self.safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]

    def prompt(self, contents):
        max_try = 4
        ii = 0
        while ii < max_try:
            try:
                response = self.client.models.generate_content(
                    model=self.version,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.0,
                        tools=[],
                        tool_config = types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode='NONE',
                            )
                        ),
                        safety_settings=self.safety_settings,
                    ),
                )
                if type(response) != str:
                    print(response)
                    if "citation_metadata" in str(response):
                        raise ValueError('Warning: Used citations by Gemini API!!!')
                    response = response.text
                return response
            except Exception as e:
                print(e)
                if ii == max_try:
                    print(response)
                    with open('errors', 'a') as f:
                        f.write(video_file)
                    continue
                else:
                    ii += 1
                    time.sleep(20)

def summarize_vl(model, video_dir, clips_dir, movie_list=None, tmstp_clips=False, source='transcripts', nb_words=1000, screenplay_path='SummScreen/screenplays'):
    """text_prompt = f'''Generate a comprehensive multimodal summary of {nb_words} words of the movie based on the provided transcript and the accompanying visual analysis (including scene descriptions, actions, and key visual elements).

Your summary should:

    Synthesize information from both the dialogue/audio (transcript) and the visual events/details (visual analysis).
    Cover the main plot points and narrative progression.
    Highlight significant character interactions and development, incorporating both dialogue and visual cues (e.g., expressions, body language, actions).
    Include descriptions of important visual moments, settings, or sequences that are crucial to understanding the story or tone, correlating them with the relevant parts of the transcript if possible.
    Present the summary in a clear, chronological order, flowing logically from beginning to end.
    Aim for a balance between textual and visual aspects to provide a holistic understanding of the movie experience.

Focus on integrating the information to show how the visual and auditory elements work together to tell the story.

Your overall summary should contain {nb_words} words.'''"""


    # remove and not more
    text_prompt = f'''Generate a comprehensive multimodal summary of exactly {nb_words} words of the movie based on the provided transcript and the most important visual elements.

Your summary should:

    Synthesize information from both the dialogue (transcript) and the important visual events (visual analysis).

Your overall summary should contain exactly {nb_words} words. Do not refer to external websites, movie databases or plot summaries.'''
    """text_prompt = f'''Generate a comprehensive summary of exactly {nb_words} words of the movie.

    Your summary should:

        Synthesize information from the given document.

    Your overall summary should contain exactly {nb_words} words. Do not refer to external websites, movie databases or plot summaries.'''"""
    if clips_dir is not None:
        out_path = f'experiments/{clips_dir.split("/")[-1]}/generations'
    elif source == 'screenplay':
        if screenplay_path == 'SummScreen/screenplays':
            out_path = 'experiments/multimodal/generations'
        else:
            out_path = f'experiments/multimodal_{screenplay_path.split("/")[-1]}/generations'
    else:
        out_path = 'experiments/monomodal/generations'
        text_prompt = f'''Generate a comprehensive summary of exactly {nb_words} words of the movie based on the provided transcript.

        Your summary should:

            Synthesize information from both the dialogue (transcript).

        Your overall summary should contain exactly {nb_words} words. Do not refer to external websites, movie databases or plot summaries.'''
    check_dir(out_path)

    for video_file in os.listdir(video_dir):
        movie_name = video_file[:-4]
        if movie_list is None or video_file[:-4] in movie_list:
            if video_file[-4:] == '.mp4' and f'{movie_name}.txt' not in os.listdir(out_path):
                print(f'>> Generating for {movie_name}')
                interleaved_context = build_interleaved_context(movie_name, clips_dir=clips_dir, tmstp_clips=tmstp_clips, source=source, screenplay_path=screenplay_path)
                print(interleaved_context)
                print(len(interleaved_context[0].split()))
                contents = interleaved_context + [text_prompt]

                response = model.prompt(contents)

                # Access the generated text response
                if response is not None:
                    with open(f'{out_path}/{movie_name}.txt', 'w') as f:
                        f.write(response)

'''def summarize_vl(self, video_dir):
    check_dir(f'experiments/end_to_end_{video_dir}')
    # Example usage
    if os.path.exists('errors'):
        with open('errors') as f:
            errors_files = f.read().split('\n')
    else:
        errors_files = []
    video_list = ['Forrest Gump_1994.mp4', 'The Sixth Sense_1999.mp4', 'The Shining_1980.mp4', 'Catch Me If You Can_2002.mp4', 'Yes Man_2008.mp4', 'Very Bad Things_1998.mp4']
    for video_file in video_list:
    #nb_videos = len(os.listdir('videos'))
    #for video_file in os.listdir('videos'):
        if video_file[-4:] == '.mp4':
            out_path = f'experiments/end_to_end_{video_dir}/{video_file[:-4]}.txt'
            if os.path.exists(out_path):
                continue
            if video_file in errors_files:
                print(f'Skip file {video_file} as Error occured with it in the past')
                continue

            video_path = f'{video_dir}/{video_file}'

            #sample_files = [PIL.Image.open(f'{image_dir}\\{image_path}') for image_path in os.listdir(image_dir)]

            prompt = 'Dialogue:\n'
            with open(f'SummScreen/transcripts/{video_file[:-4]}.json') as f:
                prompt = '\n'.join(json.load(f)['Transcript'])

            #question_prompt = 'Describe what happens in the full video.'
            question_prompt = 'Provide a precise visual description of every single clip of the video. Always refer to the provided dialogue in your description'
            #question_prompt = "Summarize every single existing subplot from the above dialogue. For each subplot, include throughout you summary any important visual detail or information about character actions, interactions, scene location that you may pick up from the provided video. Your summary should be very complete."
            prompt += f"\n\n{question_prompt}"

            response = self.prompt(video_path, prompt)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(response)'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key',type=str)
    parser.add_argument('--path',type=str)
    parser.add_argument('--nb_words',type=int,default=1000)
    parser.add_argument('--version',type=str,default='gemini-2.5-flash')
    ARGS = parser.parse_args()

    movie_list = get_movie_list(list_id=0)
    #movie_list = get_movie_list(list_id=1)
    #movie_list = ['Forrest Gump_1994', 'Legion_2010', 'Wonder Woman_2017', "I'm Thinking of Ending Things_2020"]
    #movie_list = ['There Will Be Blood_2007']
    #movie_list = None
    if 'gemini' in ARGS.version:
        prompter = PrompterVL(version=ARGS.version, key=ARGS.api_key)
    elif 'qwen' in ARGS.version:
        prompter = Qwen_next()
    summarize_vl(prompter, 'SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=ARGS.nb_words, screenplay_path=ARGS.path)

    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000)
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1150, screenplay_path='SummScreen/transcripts')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_gt_clips_gemini-2.0-flash')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_random_clips')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_selected_clips')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_agent_clips_precise')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_video_agent_clips_id_gt_50') #Before: 1050
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_video_agent_clips_id_50_qwen')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_video_agent_clips_id_50_qwen_omni')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_selected_clips_nb_clips_50')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_random_clips_nb_clips_50')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000) # repeat 3 times length constraint

    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1300, screenplay_path='SummScreen/utt')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/built_screenplay_video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_few_shot_better')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_few_shot_better')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_few_shot_better')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_video_agent_clips_id_gt_50_step_10_llm_gemini_few_shot_better')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_few_shot_better_captions_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_random_clips_nb_clips_50_qwen_omni')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_random_clips_nb_clips_50')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_random_clips_nb_clips_50_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_few_shot_better_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='transcripts', movie_list=movie_list, nb_words=1400)
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_selected_clips_nb_clips_50_exact')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_random_clips_nb_clips_50')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_few_shot_better')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_gt_clips_gemini-2.0-flash_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_selected_clips_nb_clips_75_exact')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000)
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_selected_clips_nb_clips_25_exact')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_video_agent_clips_id_25_step_5_llm_gemini_qwen_omni_local_few_shot_better_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_video_agent_clips_id_75_step_15_llm_gemini_qwen_omni_local_few_shot_better_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_few_shot_better_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_random_clips_nb_clips_50_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_gt_clips_gemini-2.0-flash_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/screenplay_gt_clips_gemini-2.0-flash')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_selected_clips_nb_clips_50_exact_tmstp_gemini')
    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, source='screenplay', movie_list=movie_list, nb_words=1000, screenplay_path='SummScreen/test_built_screenplay_video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_gemini')


    #prompter.summarize_vl('SummScreen/videos', clips_dir=None, movie_list=movie_list, nb_words=1700)
    #prompter.summarize_vl('SummScreen/videos', clips_dir='SummScreen/gt_video_clips_gemini-2.0-flash', tmstp_clips=True, movie_list=movie_list, nb_words=1200)
    #prompter.summarize_vl('SummScreen/videos', clips_dir='SummScreen/random_clips', tmstp_clips=True, movie_list=movie_list, nb_words=1400)
    #prompter.summarize_vl('SummScreen/videos', clips_dir='SummScreen/all_clips', tmstp_clips=True, movie_list=movie_list, nb_words=1200)
    #prompter.summarize_vl('SummScreen/videos', clips_dir='SummScreen/agent_clips', tmstp_clips=, movie_list=movie_list, nb_words=1200)
    #prompter.summarize_vl('SummScreen/videos', clips_dir='SummScreen/selected_clips', tmstp_clips=True, movie_list=movie_list, nb_words=1200)
