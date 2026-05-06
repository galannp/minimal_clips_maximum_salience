import os
import random
from dl_utils.misc import check_dir
from utils.movie_list import get_movie_list
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_clips',type=int,default=50)
    ARGS = parser.parse_args()

    zero_shot_clips_path = 'SummScreen/zero_shot_clips'
    out_path = f'SummScreen/random_clips_nb_clips_{ARGS.nb_clips}'
    check_dir(out_path)

    for movie_name in get_movie_list(list_id=0):
        if not os.path.exists(f'{out_path}/{movie_name}'):
            random_clips = random.sample(list(os.listdir(f'{zero_shot_clips_path}/{movie_name}')), ARGS.nb_clips)
            all_clips_tmstp = [int(a[:-4].split('_')[1]) for a in random_clips]

            with open(f'{out_path}/{movie_name}', 'w') as f:
                f.write('\n'.join(list(map(str, sorted(all_clips_tmstp)))))
