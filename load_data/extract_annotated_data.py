import os
from datasets import load_dataset
import json

def process_screenplay(screenplay, add_captions=True, remove_scene_loc=False):
    script = ''
    i = 0
    caption_prev = False
    for line in screenplay.split('\n'):
        line = line.strip()
        beg, line = line.split('>', maxsplit=1)
        line = line.split('<')[0]
        beg = beg[1:]
        if beg == 'character':
            if caption_prev:
                script += "\n"
                caption_prev = False
            script += line + ': '
        elif beg == 'dialogue':
            if caption_prev:
                script += "\n"
                caption_prev = False
            script += line + '\n'
        elif add_captions and (beg == 'scene_description' or not remove_scene_loc and beg == 'stage_direction'):
            if caption_prev:
                script += ' ' + line
            else:
                script += f'Caption {i}: {line}'
                i += 1
            caption_prev = True
    return script


def save_movie(movie_name, ds_test):
    movie = ds_test.filter(lambda x: x['movie_name'] == movie_name)
    raw_screenplay = movie['script'][0]
    summary = movie['summary'][0]

    screenplay = process_screenplay(raw_screenplay, add_captions=True)
    transcripts = process_screenplay(raw_screenplay, add_captions=False)

    with open(f'SummScreen/summaries/{movie_name}.json', 'w', encoding='utf-8') as f:
        json.dump({'soap_central': summary}, f, indent=4, ensure_ascii=False)
    with open(f'SummScreen/transcripts/{movie_name}.json', 'w', encoding='utf-8') as f:
        json.dump({'Transcript': transcripts.split('\n')}, f, indent=4, ensure_ascii=False)
    with open(f'SummScreen/screenplays/{movie_name}.json', 'w', encoding='utf-8') as f:
        json.dump({'Screenplay': screenplay.split('\n')}, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    ds_test = load_dataset("rohitsaxena/MovieSum")['test']

    for movie_name in os.listdir('SummScreen/videos'):
        if movie_name[-4:] == '.mp4':
            print(movie_name)
            save_movie(movie_name[:-4], ds_test)
