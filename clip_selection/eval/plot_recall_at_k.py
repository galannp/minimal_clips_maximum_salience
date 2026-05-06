import os
import json
from utils.movie_list import get_movie_list
import json
import screenplay_from_video_clips
import screenplay_from_zero_shot_clips
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse

def nb_captions_screenplay_orig(movie_name):
    screenplay = json.load(open(f'SummScreen/screenplays/{movie_name}.json'))['Screenplay']
    count = 0
    for line in screenplay:
        if line.startswith('Caption'):
            count += 1
    return count

def plot_curves(thresholds, curves, K):
    for key, item in curves.items():
        plt.plot(thresholds, item, label=key)

    plt.xlabel('threshold (' + r'$\tau$' + ')', fontsize=16)
    plt.ylabel(f'Recall@{K}', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.legend(labelcolor='linecolor')
    plt.legend(loc='upper right', fontsize=16)
    plt.grid(True)
    plt.savefig(f'plot_{K}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epname',nargs='+',default=['all'])
    parser.add_argument('--K',type=int)
    ARGS = parser.parse_args()

    #thresholds = np.linspace(0.7, 0, 50)
    thresholds = np.linspace(0.3, 0.3, 1)
    if ARGS.K == 25:
        clips_files = {'25 random clips': 'random_clips_nb_clips_25', 
            '25 silent clips aligned': 'selected_clips_nb_clips_25_exact_aligned_utt', 
            '25 ours clips zero-shot': 'video_agent_clips_id_25_step_5_llm_gemini_qwen_omni_local',
            '25 ours clips two-shot': 'video_agent_clips_id_25_step_5_llm_gemini_qwen_omni_local_few_shot_better','25 ours clips one-shot scenes': 'video_agent_clips_id_25_step_5_llm_gemini_qwen_omni_local_scenes', '25 ours clips two-shot scenes': 'video_agent_clips_id_25_step_5_llm_gemini_qwen_omni_local_scenes_few_shot_better','25 ours clips two-shot scenes': 'video_agent_clips_id_25_step_5_llm_gemini_qwen_omni_local_params_3_few_shot_better','25 ours clips two-shot scenes': 'video_agent_clips_id_25_step_5_llm_gemini_qwen_omni_local_params_3'}
    elif ARGS.K == 50:
        clips_files = {'50 random clips': 'random_clips_nb_clips_50', 
            '50 silent clips aligned': 'selected_clips_nb_clips_50_exact_aligned_utt', 
            '50 ours clips zero-shot': 'video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local',
            '50 ours clips two-shot': 'video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_few_shot_better','50 ours clips one-shot scenes': 'video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_scenes',
            '50 ours clips two-shot scenes': 'video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_scenes_few_shot_better','25 ours clips two-shot scenes': 'video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_params_3_few_shot_better','50 ours clips two-shot scenes': 'video_agent_clips_id_50_step_10_llm_gemini_qwen_omni_local_params_3'}
    else:
        clips_files = {'75 random clips': 'random_clips_nb_clips_75', 
            '75 silent clips aligned': 'selected_clips_nb_clips_75_exact_aligned_utt', 
            '75 ours clips zero-shot': 'video_agent_clips_id_75_step_15_llm_gemini_qwen_omni_local',
            '75 ours clips two-shot': 'video_agent_clips_id_75_step_15_llm_gemini_qwen_omni_local_few_shot_better','75 ours clips one-shot scenes': 'video_agent_clips_id_75_step_15_llm_gemini_qwen_omni_local_scenes','75 ours clips two-shot scenes': 'video_agent_clips_id_75_step_15_llm_gemini_qwen_omni_local_scenes_few_shot_better','25 ours clips two-shot scenes': 'video_agent_clips_id_75_step_15_llm_gemini_qwen_omni_local_params_3_few_shot_better','75 ours clips two-shot scenes': 'video_agent_clips_id_75_step_15_llm_gemini_qwen_omni_local_params_3'}

    curves = defaultdict(list)

    if ARGS.epname == ['all']:
        #movie_list = get_movie_list(list_id=-2)
        movie_list = get_movie_list(list_id=0)
    else:
        movie_list = ARGS.epname
    version_gemini = 'gemini-2.0-flash'
    for label, clip_file in clips_files.items():
        print(clip_file)
        for threshold in thresholds:
            avg_precision, avg_recall, avg_f1, avg_len, avg_tot = 0, 0, 0, 0, 0
            i = 0
            for movie_name in movie_list:
                gt_clips_path = f'groundtruth_clips_{version_gemini}/{movie_name}'
                if os.path.exists(gt_clips_path):
                    gt_clips = set(map(int, open(gt_clips_path).read().split('\n')))

                    video_clips_dir = f'SummScreen/{clip_file}'
                    if clip_file.startswith('selected_clips'):
                        screenplay = screenplay_from_video_clips.build_screenplay(video_clips_dir, movie_name, threshold=threshold)
                    else:
                        screenplay = screenplay_from_zero_shot_clips.build_screenplay(video_clips_dir, movie_name, threshold=threshold)
                    if screenplay is None:
                        continue
                    existing_caption_ids = set([int(line.split(':', maxsplit=1)[0].split('Caption')[1].strip()) for line in screenplay if line.startswith('Caption')])

                    matching_clips = existing_caption_ids.intersection(gt_clips)
                    if len(existing_caption_ids) > 0:
                        precision = len(matching_clips) / max(50, len(existing_caption_ids))
                        recall = len(matching_clips) / len(gt_clips)
                        if precision == recall == 0:
                            f1 = 0
                        else:
                            f1 = 2 * precision * recall / (precision + recall)

                        print(recall, movie_name, precision, f1, len(existing_caption_ids), nb_captions_screenplay_orig(movie_name), f'nb_gt_clips: {len(gt_clips)}')
                        avg_precision += precision
                        avg_recall += recall
                        avg_f1 += f1
                        avg_len += len(existing_caption_ids)
                        avg_tot += nb_captions_screenplay_orig(movie_name)
                        i += 1

            if i == 0:
                avg_recall = 0
            else:
                avg_precision /= i
                avg_recall /= i
                avg_f1 /= i
                avg_len /= i
                avg_tot /= i

            curves[label].append(avg_recall)

            print(i, avg_precision, avg_recall, avg_f1, avg_len, avg_tot)
            print('\n')

    plot_curves(thresholds, curves, ARGS.K)
