from prompt_gemini import Prompter
import argparse
import os
import pandas as pd
from utils.movie_list import get_movie_list
from dl_utils.misc import check_dir

if __name__ == '__main__':
    version_facts = 'gemini-2.0-flash'
    #version_prompt = 'gemini-2.0-flash'
    #version_prompt = 'gemini-2.5-flash-preview-04-17'
    version_prompt = 'gemini-2.5-flash-preview-05-20'
    parser = argparse.ArgumentParser()
    parser.add_argument('--epname',type=str,nargs='+', default=['all'])
    parser.add_argument('--expname',type=str)
    parser.add_argument('--api_key')
    parser.add_argument('--truncate_lim',type=int,default=None)
    parser.add_argument('--exp_path',default='experiments')
    ARGS = parser.parse_args()

    llm = Prompter(key=ARGS.api_key, version=version_prompt)

    expdir = f'{ARGS.exp_path}/{ARGS.expname}/generations'
    scores_dir = f'{ARGS.exp_path}/{ARGS.expname}'
    check_dir(scores_dir)

    if os.path.exists(f'{scores_dir}/scores.csv'):
        all_scores = pd.read_csv(f'{scores_dir}/scores.csv', index_col=0, header=0).to_dict(orient='index')
    else:
        all_scores = {}

    movie_list_list = get_movie_list(list_id=0)
    for video_file in os.listdir('SummScreen/videos'):
        if video_file[-4:] == '.mp4' and video_file[:-4] in movie_list_list and f'{video_file[:-4]}.txt' in os.listdir(expdir):
            movie_name = video_file[:-4]
            print(movie_name)
            if movie_name not in all_scores:
                summary = open(f'{expdir}/{movie_name}.txt').read()
                if ARGS.truncate_lim is not None:
                    summary = ' '.join(summary.split(' ')[:ARGS.truncate_lim])
                vision_facts = open(f'gpt-extracted-facts_{version_facts}/vision_facts/{movie_name}').read().split('\n')
                text_facts = open(f'gpt-extracted-facts_{version_facts}/text_facts/{movie_name}').read().split('\n')
                vision_facts = '\n'.join(['* ' + f for f in vision_facts])
                text_facts = '\n'.join(['* ' + f for f in text_facts])
                prompt = '''<Beginning of Summary>
    {}
    <End of Summary>

    Task:
        For each fact listed below, determine whether the exact meaning of the fact is explicitly present in the summary above.

    Instructions:
        You must justify your answer by quoting or paraphrasing the relevant part of the summary.
        If the fact is not explicitly present, even if it seems implied or suggested, you must answer No.
        Do not accept facts just because they are likely, inferable, or assumed from context.
        However, do allow for reasonable paraphrasing or rewording. If the summary conveys the same meaning as the fact using different but equivalent words, answer Yes.

    Format:

    Fact 1: [Recopy the Fact]
    1. Justification (quote or paraphrase from the summary, and explain how it matches the fact)
    2. Yes

    Fact 2: [Recopy the Fact]
    1. Justification
    2. No

    ...

    Fact N: [Recopy the Fact]
    1. Justification
    2. Yes


    <Beginning of Facts>
    {}
    <End of Facts>

        '''
                """prompt = '''<Beginning of Summary>
    {}
    <End of Summary>

    Task:
    For every fact below, can you find the fact in the summary above?

    {}

    Answer in the following way:
    1. Justify your answer
    2. Answer by Yes or No

    Example:

    Fact 1: Recopy the Fact
    1. Justification
    2. Yes

    Fact 2: Recopy the Fact
    1. Justification
    2. No

    ...

    Fact N: Recopy the Fact
    1. Justification
    2. Yes

    '''"""

                vis_rec = 0
                count_vis = 0
                out = llm.prompt([prompt.format(summary, vision_facts)], [''])[0]
                for line in out.split('\n'):
                    if line.strip().startswith('1. '):
                        if curr_fact.lower() in line.lower():
                            is_copy_paste = True
                    elif line.strip().startswith('2. '):
                        if 'Yes' in line and not is_copy_paste:
                            vis_rec += 1
                        count_vis += 1
                    elif 'Fact' in line:
                        split = line.split(':', maxsplit=1)
                        if len(split) > 1:
                            is_copy_paste = False
                            curr_fact = split[1].strip(' *')
                    else:
                        is_copy_paste = False
                        curr_fact = line.strip(' *')
                print(vis_rec)

                text_rec = 0
                count_text = 0
                out = llm.prompt([prompt.format(summary, text_facts)], [''])[0]
                for line in out.split('\n'):
                    if line.strip().startswith('1. '):
                        if curr_fact.lower() in line.lower():
                            is_copy_paste = True
                    elif line.strip().startswith('2. '):
                        if 'Yes' in line and not is_copy_paste:
                            text_rec += 1
                        count_text += 1
                    elif 'Fact' in line:
                        split = line.split(':', maxsplit=1)
                        if len(split) > 1:
                            is_copy_paste = False
                            curr_fact = split[1].strip(' *')
                print(text_rec)

                fact_rec = (vis_rec + text_rec) / (count_vis + count_text)
                vis_rec /= count_vis
                text_rec /= count_text
                mfactsum = (vis_rec + text_rec) / 2
                all_scores[movie_name] = {'visual_recall': vis_rec, 'textual_recall': text_rec, 'mfactsum': mfactsum, 'fact_recall': fact_rec, 'nb_vis_facts': count_vis, 'nb_text_facts': count_text, 'nb_facts': count_vis + count_text, 'avg_len': len(summary.split())}

                all_scores_df = pd.DataFrame(all_scores).T
                print(all_scores_df)
                print('\n\n\n')
                print(all_scores_df.describe())

                all_scores_df.to_csv(f'{scores_dir}/scores.csv')

    all_scores_df = pd.DataFrame(all_scores).T
    print(all_scores_df)
    print('\n\n\n')
    print(all_scores_df.describe())