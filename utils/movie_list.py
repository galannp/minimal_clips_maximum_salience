import json
import os

def get_movie_list(list_id):
    l = [vv[:-4] for vv in os.listdir('SummScreen/videos')]
    final_l = []
    for ll in l:
        s = json.load(open(f'SummScreen/screenplays/{ll}.json'))['Screenplay']
        if len([line for line in s if line.startswith('Caption')]) > 50:
            final_l.append(ll)
        else:
            print(f'Removing {ll}')
    print(f'Test set: {len(final_l)} movies')
    return final_l
