import os, json
from dl_utils.misc import check_dir
from utils.movie_list import get_movie_list
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--clip_selection',type=str,choices=['random', 'silent', 'ours'], default='ours')
parser.add_argument('--nb_clips',type=int,default=50)
parser.add_argument('--recaptioning',action='store_true')
parser.add_argument('--few_shot_examples',action='store_true')
ARGS = parser.parse_args()


check_bad_numbers_captions=False
aligned_path = 'SummScreen/aligned_transcripts'

if ARGS.recaptioning:
    caption_model = 'gemini-2.5-flash-lite'
else:
    caption_model = 'qwen_omni'

if ARGS.clip_selection == 'random':
    captions_tmstp_file = f'random_clips_nb_clips_{ARGS.nb_clips}'
    captions_path = f'zero_shot_clips_{caption_model}_captions'
elif ARGS.clip_selection == 'silent':
    captions_tmstp_file = f'selected_clips_nb_clips_{ARGS.nb_clips}_exact_tmstp'
    captions_path = f'selected_clips_{caption_model}_captions'
else:
    captions_tmstp_file = f'video_agent_clips_id_{ARGS.nb_clips}_step_10_llm_gemini_qwen_omni_local'
    if ARGS.few_shot_examples:
        captions_tmstp_file += '_few_shot_better'
    captions_path = f'zero_shot_clips_{caption_model}_captions'
captions_tmstp_path = f'SummScreen/{captions_tmstp_file}'

if ARGS.recaptioning:
    out_path = f'SummScreen/test_built_screenplay_{captions_tmstp_file}_gemini'
else:
    out_path = f'SummScreen/test_built_screenplay_{captions_tmstp_file}_qwen_omni_local'
check_dir(out_path)

def get_captions(captions_path, truncate=False):
    captions = open(f'{captions_path}/{movie_name}.txt').read()
    new_caption = []
    all_captions = {}
    theoretical_caption_tmstp = 10000
    for a in captions.split('\n'):
        if a.startswith('Caption'):
            if new_caption != []:
                e = '\n'.join(new_caption)
                if truncate:
                    e = ' '.join(e.split()[:200])
                all_captions[caption_tmstp] = e
                new_caption = []
            caption_tmstp = int(a.split(':', maxsplit=1)[0].split('Caption', maxsplit=1)[1].strip())
            if check_bad_numbers_captions and not (caption_tmstp == theoretical_caption_tmstp or caption_tmstp % 20000 != 10000):
                print(caption_tmstp)
                exit()
            theoretical_caption_tmstp += 20000
        new_caption.append(a)
    if new_caption != []:
        e = '\n'.join(new_caption)
        if truncate:
            e = ' '.join(e.split()[:200])
        all_captions[caption_tmstp] = e
    if check_bad_numbers_captions and not (caption_tmstp == (len(list(all_captions.values())) - 1) * 20000 + 10000 or caption_tmstp % 20000 != 10000):
        print(caption_tmstp)
        exit()
    return all_captions

for movie_name in get_movie_list(list_id=0):
    print(movie_name)
    aligned_utt = json.load(open(f'{aligned_path}/{movie_name}.json'))['Transcript']
    if os.path.exists(f'{captions_tmstp_path}/{movie_name}'):
        captions_tmstp = list(map(int, open(f'{captions_tmstp_path}/{movie_name}').read().split('\n')))
    elif os.path.exists(f'{captions_tmstp_path}/{movie_name}.npy'):
        captions_tmstp = map(int, list(np.mean(np.load(f'{captions_tmstp_path}/{movie_name}.npy'), axis=1)))
    captions = get_captions(captions_path, truncate=True)

    built_screenplay = []
    curr_line = aligned_utt[0]
    i = 0
    for cap in captions_tmstp:
        while i < len(aligned_utt) and curr_line[1] < cap:
            curr_line = aligned_utt[i]
            built_screenplay.append(curr_line[2])
            i += 1
        #if cap not in captions:
        #    print(movie_name, cap)
        else:
            built_screenplay.insert(-1, captions[cap])
    while i < len(aligned_utt):
        curr_line = aligned_utt[i]
        built_screenplay.append(curr_line[2])
        i += 1

    json.dump({'Screenplay': built_screenplay}, open(f'{out_path}/{movie_name}.json', 'w'), indent=4)
