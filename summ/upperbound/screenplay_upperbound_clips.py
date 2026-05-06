import os, json
import random
from dl_utils.misc import check_dir
import argparse
from utils.movie_list import get_movie_list

def get_captions(captions_path):
    captions = open(captions_path).read()
    new_caption = []
    all_captions = {}
    theoretical_caption_tmstp = 10000
    for a in captions.split('\n'):
        if a.startswith('Caption'):
            if new_caption != []:
                e = '\n'.join(new_caption)
                all_captions[caption_tmstp] = e
                new_caption = []
            caption_tmstp = int(a.split(':', maxsplit=1)[0].split('Caption', maxsplit=1)[1].strip())
            theoretical_caption_tmstp += 20000
        new_caption.append(a)
    if new_caption != []:
        e = '\n'.join(new_caption)
        all_captions[caption_tmstp] = e
    return all_captions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_times',type=int,default=1)
    parser.add_argument('--nb_max',type=int,default=None)
    parser.add_argument('--clip_path',type=str,default='groundtruth_clips_gemini-2.0-flash')
    parser.add_argument('--do_random',action='store_true')
    parser.add_argument('--clips_caption_path',type=str,default=None)
    parser.add_argument('--epname',choices=['all', 'movie_list'],default='all')
    ARGS = parser.parse_args()

    if ARGS.epname == 'all':
        video_list = os.listdir('SummScreen/videos')
    elif ARGS.epname == 'movie_list':
        video_list = [f'{movie_name}.mp4' for movie_name in get_movie_list(list_id=0)]

    for video_file in video_list:
        movie_name = video_file[:-4]
        print(movie_name)
        if ARGS.clips_caption_path is not None:
            all_captions = get_captions(f'{ARGS.clips_caption_path}/{movie_name}.txt')
        gt_clips_path = f'{ARGS.clip_path}/{movie_name}'

        if ARGS.clip_path == 'groundtruth_clips_gemini-2.0-flash':
            if ARGS.clips_caption_path is None:
                out_path_gt = f'SummScreen/screenplay_gt_clips_gemini-2.0-flash'
            else:
                out_path_gt = f'SummScreen/test_built_screenplay_gt_clips_gemini-2.0-flash_gemini'
        else:
            out_path_gt = f'SummScreen/screenplay_{ARGS.clip_path.split("/")[-1]}'
        check_dir(out_path_gt)

        if ARGS.nb_max is not None:
            out_path_random = f'SummScreen/screenplay_random_clips_nb_clips_{ARGS.nb_max}'
        elif ARGS.nb_times == 1:
            out_path_random = f'SummScreen/screenplay_random_clips'
        else:
            out_path_random = f'SummScreen/screenplay_random_clips_times_{ARGS.nb_times}'
        #check_dir(out_path_random)
        if os.path.exists(gt_clips_path):
            gt_clips = open(gt_clips_path).read().split('\n')
            aligned_screenplay = json.load(open(f'SummScreen/aligned_screenplays/{movie_name}.json'))['Transcript']

            if not os.path.exists(f'{out_path_gt}/{movie_name}.json'):
                screenplay_gt_clips = []
                ii = 0
                for line in aligned_screenplay:
                    if line[2].startswith('Caption'):
                        if line[2].split(':', maxsplit=1)[0].split('Caption')[1].strip() in gt_clips:
                            if ARGS.clips_caption_path is None:
                                screenplay_gt_clips.append(line[2])
                            else:
                                tmstp = int((line[0] / 1000 + line[1] / 1000) / 2 * 1000)
                                if tmstp in all_captions:
                                    screenplay_gt_clips.append(all_captions[tmstp])
                                else:
                                    print(tmstp)
                            ii += 1
                    else:
                        screenplay_gt_clips.append(line[2])

                with open(f'{out_path_gt}/{movie_name}.json', 'w') as f:
                    json.dump({'Screenplay': screenplay_gt_clips}, f)
                print(movie_name, ii)


            '''if ARGS.do_random:
                if not os.path.exists(f'{out_path_random}/{movie_name}.json'):
                    nb_gt_clips = len(gt_clips)
                    for line in screenplay[::-1]:
                        if line.startswith('Caption'):
                            max_caption = int(line.split(':', maxsplit=1)[0].split('Caption')[1].strip())
                            break

                    if ARGS.nb_max is not None:
                        random_clips_id = random.sample(range(0, max_caption + 1), min(max_caption, ARGS.nb_max))
                    else:
                        random_clips_id = random.sample(range(0, max_caption + 1), nb_gt_clips * ARGS.nb_times)

                    screenplay_random_clips = []
                    for line in screenplay:
                        if line.startswith('Caption'):
                            if int(line.split(':', maxsplit=1)[0].split('Caption')[1].strip()) in random_clips_id:
                                screenplay_random_clips.append(line)
                        else:
                            screenplay_random_clips.append(line)

                    with open(f'{out_path_random}/{movie_name}.json', 'w') as f:
                        json.dump({'Screenplay': screenplay_random_clips}, f)'''
