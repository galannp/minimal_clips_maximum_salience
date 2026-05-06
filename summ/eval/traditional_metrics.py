import json
import numpy as np
from dl_utils.misc import set_experiment_dir
from dl_utils.misc import check_dir
from utils.utils import rouge_from_multiple_refs, meteor_from_multiple_refs, bertscore_from_multiple_refs
from datasets import load_dataset, load_from_disk

import argparse
import torch
import os
from os.path import join
import sys
from utils.utils import get_fn, display_rouges, display_bertscores
from tqdm import tqdm
from utils.movie_list import get_movie_list

def summarize_screenplays(screenplays, batch_size, prompt_type, llm='llama', only_visual=False):
    if llm == 'llama':
        from prompt_llama import Prompter
        model = Prompter(batch_size=batch_size, input_chunk_size=128000, max_new_tokens=1024)
    elif llm == 'qwen':
        from prompt_qwen import Prompter
        model = Prompter()
    else:
        from prompt_gemini import Prompter
        model = Prompter()
    if only_visual:
        summarization_question = "Combine all 'Video Description' from the above into one coherent text. Focus only in combining every above 'Video Description' into on coherent document. Always explicitely state the name of the characters in your output."
        summaries = model.prompt_llama(screenplays, summarization_question)
        summarization_question = 'Rewrite your output in no more than 500 words.'
        summaries = model.prompt_llama(summaries, summarization_question)
    else:
        #summarization_question = 'Provide a complete summary of the above. Here is a highly detailed summary:'
        #summarization_question = 'Provide a summary of the above video Caption and dialogue. Write a summary that merges the information from the video Caption together with the dialogue into one coherent paragraph. Please match the information from the video Caption with the one from the dialogue. Explicitly state the name of any character appearing in the captions. Here is the summary:'
        if prompt_type == 'custom':
            summarization_question = 'Summarize all existing subplots from the above dialogue. For each subplot, include throughout you summary any important visual detail or information about character actions, interactions, scene location that you may find in the video Captions.'
        elif prompt_type == 'custom_600':
            summarization_question = 'Summarize all existing subplots from the above dialogue. For each subplot, include throughout you summary any important visual detail or information about character actions, interactions, scene location that you may find in the video Captions. Make sure your summary does not exceed 600 words.'
        elif prompt_type == 'gemini_screenplay':
            summarization_question = "Summarize every single existing subplots from the above dialogue. For each subplot, include throughout you summary every single visual detail or important information you can pick up from the video captions. Always explicitely connect the visual information from the visual Captions to the story. The video captions are given to you in the above dialogue by 'Caption:'. Include the information from all video captions into your summary. Always explicitely state the name of the characters mentionned in the video captions. Always refer to the video captions by rephrasing them but without quoting them. Your overall summary should not exceed 500 words"
        elif prompt_type == 'default_long':
            summarization_question = "Summarize every single existing subplots from the above dialogue in every single detail. Your summary should be very complete with respect to the above transcripts."
        else:
            #summarization_question = 'Provide a complete summary of the above. Here is a detailed summary:'
            #summarization_question = 'Provide a complete summary of the above in 400 to 600 words. Here is a precise and detailed summary:'
            summarization_question = "Summarize every single existing subplots from the above dialogue. Your overall summary should not exceed 500 words"
        #summarization_question = 'Summarize the plot points of the above dialogue. For each subplot, include throughout you summary any important visual detail or information about character actions, interactions, scene location that you may find in the video Captions.'
        summaries = model.prompt_llama(screenplays, summarization_question)
    return summaries

def evaluate(summaries, references, episode_names, expdir=None):
    if expdir is not None:
        check_dir(expdir)
        res_f = open(join(expdir,'results.txt'),'w')

    if True:
        if expdir is not None:
            res_f.write('\nTEST ROUGES:\n')
            res_f.write(f'\nAll test rouges:\n')

            check_dir(generations_dir := join(expdir, 'generations'))

        print('Pipeline Evaluation')
        rouges = []

        for summary, refs, en in zip(summaries, references, episode_names):
            best_rouge = rouge_from_multiple_refs(summary, refs, return_full=False, benchmark_rl=True)

            rouges.append(best_rouge)

            if expdir is not None:
                res_f.write(f'rouge for {en}: {best_rouge[0]:.3f} {best_rouge[1]:.3f} {best_rouge[2]:.3f} {best_rouge[3]:.3f}\n')
                #with open(f'{generations_dir}/{en}.txt','w') as gen_f:
                #    gen_f.write(summary)

        print(rouges)
        rouges = np.array(rouges).mean(axis=0)
        print(disp_rouges := display_rouges(rouges))

        if expdir is not None:
            res_f.write('\n\n')
            for rname,rscore in disp_rouges:
                res_f.write(f'{rname}: {rscore:.5f}\n')


    if True:
        if expdir is not None:
            res_f.write('\nTEST METEOR:\n')
            res_f.write(f'\nAll test meteor:\n')

        meteors = []

        for summary, refs, en in zip(summaries, references, episode_names):
            best_meteor = meteor_from_multiple_refs(summary, refs)

            meteors.append(best_meteor)

            if expdir is not None:
                res_f.write(f'meteor for {en}: {best_meteor:.3f}\n')

        meteors = np.array(meteors).mean(axis=0)
        print(disp_meteors := [('meteor', meteors)])

        if expdir is not None:
            res_f.write('\n\n')
            for rname,rscore in disp_meteors:
                res_f.write(f'{rname}: {rscore:.5f}\n')


    if False:
        if expdir is not None:
            res_f.write('\nTEST BERTSCORE:\n')
            res_f.write(f'\nAll test bertscores:\n')

        bertscores = []

        for summary, refs, en in zip(summaries, references, episode_names):
            best_bertscore = bertscore_from_multiple_refs(summary, refs)

            bertscores.append(best_bertscore)

            if expdir is not None:
                res_f.write(f'bertscore for {en}: {best_bertscore[0]:.3f} {best_bertscore[1]:.3f} {best_bertscore[2]:.3f}\n')

        bertscores = np.array(bertscores).mean(axis=0)
        print(disp_bertscores := display_bertscores(bertscores))

        if expdir is not None:
            res_f.write('\n\n')
            for rname,rscore in disp_bertscores:
                res_f.write(f'{rname}: {rscore:.5f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epname',type=str,nargs='+', default=['movie_list'])
    parser.add_argument('--show_name',type=str, default='all')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--prompt_type', type=str, choices=['custom', 'default', 'gemini_screenplay', 'default_long', 'custom_600'], default='custom')
    parser.add_argument('--expdir_prefix',type=str,default='experiments')
    parser.add_argument('--screenplay_dir',type=str)
    parser.add_argument('--expname',type=str)
    parser.add_argument('--regenerate',action='store_true')
    parser.add_argument('--llm', type=str,choices=['llama', 'gemini', 'qwen'], default='llama')
    parser.add_argument('--only_visual', action='store_true')
    parser.add_argument('--truncate_length',type=int,default=None)
    ARGS = parser.parse_args()

    if ARGS.expname is None:
        sys.exit('must set explicit expname')
    expdir = join(ARGS.expdir_prefix, ARGS.expname)
    set_experiment_dir(expdir, overwrite=True, name_of_trials=join(ARGS.expdir_prefix, 'tmp'))

    if ARGS.epname == ['all']:
        all_epnames = [fn[:-4] for fn in os.listdir('SummScreen/videos')]
    elif ARGS.epname == ['all_existing']:
        all_epnames = [fn[:-4] for fn in os.listdir(f'{ARGS.expdir_prefix}/{ARGS.expname}/generations')]
    elif ARGS.epname == ['movie_list']:
        all_epnames = get_movie_list(list_id=0)
    else:
        all_epnames = ARGS.epname

    if ARGS.show_name != 'all':
        all_epnames = [x for x in all_epnames if x.startswith(ARGS.show_name)]

    # Debug
    #all_epnames = ['oltl-08-19-09', 'oltl-09-23-08', 'gl-02-25-05', 'atwt-01-04-05', 'gl-02-10-05', 'oltl-03-01-10', 'atwt-01-21-05', 'gl-01-06-05', 'gl-01-31-05', 'bb-05-19-06', 'gl-02-24-05', 'oltl-07-10-09', 'bb-05-29-06', 'bb-01-26-07', 'gl-03-02-05', 'atwt-01-20-03', 'atwt-01-13-05', 'oltl-01-08-09', 'gl-01-27-05', 'atwt-01-11-05', 'gl-03-01-06', 'bb-04-27-17', 'gl-02-14-05', 'bb-06-12-06', 'pc-05-29-03', 'atwt-01-16-07', 'bb-01-16-07', 'pc-04-03-03', 'gl-02-09-05', 'atwt-01-05-10', 'atwt-11-11-05', 'bb-05-15-06', 'oltl-09-24-08', 'oltl-06-29-09', 'atwt-05-15-06', 'atwt-05-19-03', 'atwt-07-25-06', 'atwt-05-12-03', 'atwt-03-09-06', 'oltl-01-26-10', 'atwt-01-05-05', 'oltl-06-02-09', 'oltl-03-10-10', 'oltl-02-05-09', 'gl-02-03-05', 'atwt-09-13-06', 'oltl-08-24-09', 'oltl-07-30-08', 'oltl-09-18-09', 'atwt-05-18-06', 'pc-05-30-03', 'atwt-09-14-06', 'bb-05-22-06', 'atwt-09-12-06', 'pc-05-13-03', 'pc-06-30-03', 'atwt-01-22-07', 'gl-01-25-05', 'gl-03-07-05', 'oltl-02-04-10', 'gl-02-21-05', 'bb-01-20-15', 'bb-01-02-07', 'oltl-07-21-11', 'bb-05-31-06', 'gl-01-12-05', 'oltl-08-04-09', 'oltl-06-24-09', 'atwt-01-18-06', 'bb-06-09-06', 'atwt-01-10-06', 'bb-06-21-06', 'pc-03-04-03', 'gl-01-10-05', 'gl-09-11-06', 'bb-05-08-17', 'oltl-06-15-09', 'bb-01-30-07', 'oltl-08-14-09', 'oltl-03-02-10', 'pc-05-06-03', 'atwt-01-11-07', 'oltl-01-06-10', 'oltl-03-17-09', 'oltl-01-25-10', 'bb-06-01-06', 'pc-05-02-03', 'atwt-01-21-03', 'oltl-01-15-09', 'atwt-01-18-05', 'gl-02-28-06', 'oltl-09-04-09', 'oltl-02-04-09', 'atwt-03-28-06', 'atwt-05-15-03', 'atwt-01-06-05', 'oltl-02-12-09', 'oltl-07-20-11', 'atwt-01-21-04', 'bb-01-11-07', 'bb-05-16-06', 'oltl-05-24-07', 'bb-05-08-06', 'bb-05-03-06', 'atwt-05-13-09', 'gl-01-05-05', 'bb-05-24-06', 'gl-09-06-06', 'atwt-01-09-06']

    i = 0
    gt_summaries = []
    while i < len(all_epnames):
        with open(os.path.join('SummScreen/summaries',f'{all_epnames[i]}.json')) as f:
            summaries_i = list(json.load(f).values())
            if len(summaries_i) == 0:
                all_epnames.pop(i)
            else:
                gt_summaries.append(summaries_i)
                i += 1

    summaries = {}
    i = 0
    while i < len(all_epnames):
        summary_en_path = f'{ARGS.expdir_prefix}/{ARGS.expname}/generations/{all_epnames[i]}.txt'
        if not os.path.exists(summary_en_path):
            print(f'Summary does not exist for {all_epnames[i]}')
            all_epnames.pop(i)
        else:
            with open(summary_en_path, encoding='latin-1') as f:
                s = f.read()
                if ARGS.truncate_length is not None:
                    s = ' '.join(s.split(' ')[:ARGS.truncate_length])
                summaries[all_epnames[i]] = s
            i += 1

    rouges = evaluate(list(summaries.values()), gt_summaries, list(summaries.keys()), expdir=join(ARGS.expdir_prefix, ARGS.expname))

    summary_path = join(expdir,'summary.txt')
    with open(summary_path,'w') as f:
        f.write(f'Expname: {ARGS.expname}\n')
