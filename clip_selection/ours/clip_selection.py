import cv2
import PIL.Image
import os
import json
import shutil
import math
from openai import OpenAI
import base64
from dl_utils.misc import check_dir
from prompt_gemini import Prompter
from utils.movie_list import get_movie_list
from time import sleep
import argparse
import random
import re

def sort_captions(captions, truncate=False, nb_div=None):
    new_caption = []
    all_captions = []
    for a in captions.split('\n'):
        if a.startswith('Caption') and new_caption != []:
            e = '\n'.join(new_caption)
            if truncate:
                e = ' '.join(e.split()[:200])
            all_captions.append(e)
            new_caption = []
        new_caption.append(a)
    if new_caption != []:
        e = '\n'.join(new_caption)
        if truncate:
            e = ' '.join(e.split()[:200])
        all_captions.append(e)
    #random.shuffle(all_captions)
    if nb_div is not None:
        nb_clips_per_step = len(all_captions) // nb_div
        all_captions = [all_captions[i:min(len(all_captions), i+nb_clips_per_step)] for i in range(0, len(all_captions), nb_clips_per_step)]
        if len(all_captions) % nb_div != 0:
            all_captions[-2] = all_captions[-2] + all_captions[-1]
            all_captions = all_captions[:-1]
    else:
        all_captions = [all_captions]
    return all_captions

parser = argparse.ArgumentParser()
parser.add_argument('--llm_upperbound',action='store_true')
parser.add_argument('--nb_clips',type=int,default=50)
parser.add_argument('--ep_list',type=str,nargs='+',default=['all'])
parser.add_argument('--nb_per_step',type=int,default=10)
parser.add_argument('--api_key',type=str)
parser.add_argument('--few_shot_examples',action='store_true')
ARGS = parser.parse_args()

clips_caption_path = 'zero_shot_clips_qwen_omni_captions'

prompter = Prompter(version='gemini-2.0-flash', key=ARGS.api_key)


if ARGS.llm_upperbound:
    out_dir = f'SummScreen/video_agent_clips_id_gt_{ARGS.nb_clips}_step_{ARGS.nb_per_step}_llm_gemini'
else:
    if 'qwen_omni' in clip_captions_path:
        model = 'qwen_omni'
    else:
        model = 'qwen'
    out_dir = f'SummScreen/video_agent_clips_id_{ARGS.nb_clips}_step_{ARGS.nb_per_step}_llm_gemini_{model}_local'
    if 'scenes' in clip_captions_path:
        out_dir += '_scenes'
    if 'params' in clip_captions_path:
        out_dir += f'_params{clip_captions_path.split("params")[1]}'
if ARGS.few_shot_examples:
    out_dir += '_few_shot_better'
check_dir(out_dir)
for movie_name in get_movie_list(list_id=0):
    if movie_name not in os.listdir(out_dir):
        print(movie_name)
    if movie_name not in os.listdir(out_dir) and (ARGS.ep_list == ['all'] or movie_name in ARGS.ep_list or ARGS.ep_list == ['all_existing'] and os.path.exists(f'{clip_captions_path}/{movie_name}.txt')):
        f = open(f'{out_dir}/{movie_name}', 'w')
        nb_div = ARGS.nb_clips // ARGS.nb_per_step
        if ARGS.llm_upperbound:
            all_input_captions = '\n'.join([line for line in json.load(open(f'SummScreen/screenplays/{movie_name}.json'))['Screenplay'] if line.startswith('Caption')])
            all_input_captions = sort_captions(all_input_captions, nb_div=nb_div)
        else:
            all_input_captions = open(f'{clip_captions_path}/{movie_name}.txt').read()
            all_input_captions = sort_captions(all_input_captions, truncate=True, nb_div=nb_div)
        su = open(f'experiments/multimodal_transcripts/generations/{movie_name}.txt').read()
        all_all_captions_numbers = []
        for captions_step in all_input_captions:
            if ARGS.few_shot_examples:
                prompt_2 = f'''Here are captions from the movie Forrest Gump_1994:

Caption 1110000: In the video, a man and woman sit on a bench in a park. The man is wearing a suit and tie while the woman wears casual clothes. They appear to be reading books together as they sit side by side. The man then turns his attention towards the woman and starts talking about something. He mentions that life is like a box of chocolates and you never know what you're going to get. He also comments on how comfortable her shoes must be and suggests she could walk all day in them.

Caption 1130000: Forrest is sitting on a bench outside. He then sits inside a doctor's office with his legs up on the table. The doctor removes Forrest's leg braces and asks him to stand up. Forrest stands up and walks around the room.

Caption 1150000: The dialogue reveals that the woman is explaining the origin of the character\'s name "Forrest Gump." She mentions that the "Forrest" part of the name comes from an incident where they were related to someone who started a club called the Ku Klux Klan. The woman explains that the "Gump" part of the name was given because sometimes people do things that don\'t make sense.

Caption 1170000: The video shows a group of boys chasing Forrest Gump as he runs down a dirt road. The boys are shouting at him to run faster, while Forrest continues to run without looking back. One of the boys falls over, but gets up quickly and continues chasing Forrest. The other boys also catch up with Forrest and start to chase him more aggressively. As they get closer, one of the boys throws a rock at Forrest, who ducks to avoid it. Another boy tries to kick him, but misses. The boys continue to chase Forrest until he reaches his home, where his mother is waiting for him. She tells him that miracles happen every day, and that some people may not believe them, but they still exist.

Caption 1190000: The man is running on the field, and he jumps over the fence. He runs to the football field and throws the ball. The coaches are watching him.

Caption 1210000: The video shows a scene where a woman holding a baby sits on a bench next to another woman who is reading a book. A man in a suit is sitting on the other side of the bench with his suitcase beside him. The woman with the baby stands up and walks away from the bench while talking to the man. She then sits back down on the bench and continues talking to him. In the background, there is a bus passing by. The dialogue includes the woman asking if the bus is the number nine, but the man corrects her and says it's the number four. They also have a conversation about someone named Wallace getting shot while they were in college.

Caption 1230000: The video shows a woman reading a book to her son on their bed. The boy asks his mother about vacation, and she explains that it is when someone goes somewhere and never comes back.

What are the 3 most important Captions that describe important action or visual event you would include in a Summary of the movie Forrest Gump_1994?
Provide your answer in the following way:
1. Caption caption_number: Justification why the Caption describes crucial action for the summary
2. Caption caption_number: Justification why the Caption describes crucial action for the summary
3. Caption caption_number:Justification why the Caption describes crucial action for the summary

Answer:
Caption 1130000: Justification: This caption depicts the removal of Forrest's leg braces, a pivotal moment signifying his physical transformation and newfound freedom. 
Caption 1170000: Justification: This caption illustrates the bullying Forrest faces and his eventual discovery of his running ability, a recurring motif in the film.
Caption 1190000: Justification: This caption depicts Forrest's accidental entry into the world of football, showcasing his unexpected athletic talent.


Here are captions from the movie Wonder Woman_2017:

Caption 4210000: The scene opens with a man sitting at his desk, looking at his watch. He then turns to face another man standing before him. The man in uniform speaks to the other man, telling him that he will do nothing. The man in uniform then walks away as the other man looks on. The scene ends with the man in uniform walking out of the room.

Caption 4230000: Diana and Steve are walking down the stairs. Steve is talking to Diana. Steve is angry at Diana for not fighting back against Ares. He tells her that she didn't stand her ground because there was no chance of changing Ares' mind. He also tells her that millions of people will die if they don't fight back. He tells her that his people are next. Summary: Steve is angry at Diana for not fighting back against Ares. He tells her that she didn't stand her ground because there was no chance of changing Ares' mind. He also tells her that millions of people will die if they don't fight back. He tells her that his people are next.

Caption 4250000: The video shows a man sitting on a chair in a room. A bomb is thrown into the room and explodes. The man gets up and runs out of the door. He then talks to another man who is standing outside the door. The man inside the room is coughing and choking on smoke.

What are the 1 most important Captions that describe important action or visual event you would include in a Summary of the movie Wonder Woman_2017?
Provide your answer in the following way:
1. Caption caption_number: Justification why the Caption describes crucial action for the summary

Answer:
Caption 4250000: Justification: This caption depicts a sudden and violent attack, showcasing the dangers faced by the characters and the chaos of the war. It emphasizes the element of surprise and the characters' ability to react quickly to threats. Therefore the Caption depicts important visual action of event.



Here are captions from the movie {movie_name}:

{captions_step}

What are the {ARGS.nb_per_step} most important Captions that describe important action or visual event you would include in the existing Summary of the movie {movie_name}?
Provide your answer in the following way:
1. Caption caption_number: Justification why the Caption describes crucial action for the summary
2. Caption caption_number: Justification why the Caption describes crucial action for the summary

...

{ARGS.nb_per_step}. Caption caption_number: Justification why the Caption describes crucial action for the summary

Answer:'''
            else:
                """prompt_2 = f'''Here are captions from the movie {movie_name}:

{captions_step}

What are the {ARGS.nb_per_step} most important Captions that describe important action or visual event you would include in the existing Summary of the movie {movie_name}?
Provide your answer in the following way:
1. Caption start_end: Justification why the Caption describes crucial action for the summary
2. Caption start_end: Justification why the Caption describes crucial action for the summary

...

{ARGS.nb_per_step}. Caption start_end: Justification why the Caption describes crucial action for the summary

Answer:'''"""
                prompt_2 = f'''Here are captions from the movie {movie_name}:

{captions_step}

What are the {ARGS.nb_per_step} most important Captions that describe important action or visual event you would include in the existing Summary of the movie {movie_name}?
Provide your answer in the following way:
1. Caption caption_number: Justification why the Caption describes crucial action for the summary
2. Caption caption_number: Justification why the Caption describes crucial action for the summary

...

{ARGS.nb_per_step}. Caption caption_number: Justification why the Caption describes crucial action for the summary

Answer:'''
            out = prompter.prompt([prompt_2], [''])[0]

            all_captions_numbers = []
            for line in out.split('\n'):
                if '.' in line:
                    line = line.split('.', maxsplit=1)[1].strip(' *')
                    if line.startswith('Caption') and ':' in line:
                        all_captions_numbers.append(re.sub(r"[^\d_]", "", line.split(':', maxsplit=1)[0].split('Caption', maxsplit=1)[1])) #.strip(', *Nnumber')
            all_all_captions_numbers.extend(sorted(all_captions_numbers, key=int))

        print(all_all_captions_numbers)
        f.write('\n'.join(all_all_captions_numbers))

'''Provide all the important clips that deserve a special attention according to you for movie understanding (example specific event or action).

Example:
Caption tmstp:
Caption tmstp:
Caption tmstp:
...'''
