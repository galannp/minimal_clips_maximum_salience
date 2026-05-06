import numpy as np
import pandas as pd
from time import time
import os
from dl_utils.misc import check_dir
import json
from nltk import word_tokenize
from difflib import SequenceMatcher
from dtw import dtw
import argparse
import subprocess as sp


N_WITHOUT_CAPTIONS = 0

def clean(line):
    if ':' not in line:
        return line
    else:
        return line.split(':')[1].lower().strip()

def cc_clean(line):
    return line.replace('[ __ ] ','').strip()

def align(xlines,ylines):
    dist_mat_ = []
    sm = SequenceMatcher()
    for xl in xlines:
        if len(xl)==0:
            dist_mat_.append([1]*len(ylines))
        else:
            sm.set_seq2(xl)
            new = []
            for yl in ylines:
                if len(yl)==0:
                    new.append(1)
                else:
                    sm.set_seq1(yl)
                    ratio = sm.find_longest_match()[2]/min(len(xl),len(yl))
                    new.append(1 - ratio)
            dist_mat_.append(new)

    dist_mat = np.stack(dist_mat_)

    alignment = dtw(dist_mat)
    return alignment, dist_mat

def millisecs_from_timestamp(timestamp):
    hrs,mins,secs_ = timestamp.split(':')
    secs, msecs = secs_.split(',')
    return int(1000 * (3600*float(hrs) + 60*float(mins) + float(secs)) + float(msecs))

def align_transcripts_and_captions(epname, closed_captions_dir, transcripts_dir):
    global N_WITHOUT_CAPTIONS
    compute_start_time = time()
    with open(f'{transcripts_dir}/{epname}.json') as f:
        cont = json.load(f)
        if 'Transcript' in cont:
            raw_transcript_lines = cont['Transcript']
        else:
            raw_transcript_lines = cont['Screenplay']

    with open(f'{closed_captions_dir}/{epname}.json') as f:
        closed_captions = json.load(f)

    transcript_lines = [word_tokenize(clean(line)) for line in raw_transcript_lines]
    if 'captions' not in closed_captions.keys():
        print(f'Can\'t split {epname}, no captions')
        N_WITHOUT_CAPTIONS += 1
        return

    cc_lines = [word_tokenize(cc_clean(x[1])) for x in closed_captions['captions']]

    cc_timestamps = [x[0] for x in closed_captions['captions']]
    starts, ends = zip(*[t.split(" --> ") for t in cc_timestamps])
    starts = [millisecs_from_timestamp(t) for t in starts]
    ends = [millisecs_from_timestamp(t) for t in ends]

    alignment, dist_mat = align(transcript_lines, cc_lines)
    return alignment, raw_transcript_lines, closed_captions['captions'], starts, ends, dist_mat

def align_transcripts(epname, closed_captions_dir, aligned_transcripts_dir, transcripts_dir):
    alignment, transcript_lines, _, start_timestamps, end_timestamps, _ = align_transcripts_and_captions(epname, closed_captions_dir, transcripts_dir)
    starts, ends = np.array(start_timestamps), np.array(end_timestamps)

    aligned_transcripts_df = pd.DataFrame({"aligned_transcript": alignment.index1, "starts": starts[alignment.index2], "ends": ends[alignment.index2]})
    aligned_transcripts_df = aligned_transcripts_df.groupby("aligned_transcript").agg({"starts": "min", "ends": "max"})

    aligned_transcripts_df["transcript"] = transcript_lines
    aligned_transcripts_list = aligned_transcripts_df.values.tolist()

    check_dir(aligned_transcripts_dir)
    aligned_transcripts_fname = f'{aligned_transcripts_dir}/{epname}.json'
    with open(aligned_transcripts_fname, 'w') as f:
        json.dump({"Transcript": aligned_transcripts_list}, f, indent=4, ensure_ascii=False)
    print(f'Aligned Transcripts saved for {epname}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--closed_captions_dir',type=str)
    parser.add_argument('--aligned_transcripts_dir',type=str)
    parser.add_argument('--transcripts_dir',type=str)
    parser.add_argument('--print_full_aligned',action='store_true')
    parser.add_argument('--epname',type=str,nargs='+',default=['all'])
    ARGS = parser.parse_args()

    if ARGS.epname == ['all']:
        all_epnames = [fn[:-4] for fn in os.listdir('SummScreen/videos') if fn.endswith('.mp4') and f'{fn[:-4]}.json' in os.listdir(ARGS.closed_captions_dir)]
        for en in all_epnames:
            if not (os.path.exists(f'{ARGS.aligned_transcripts_dir}/{en}.json')):
                print(f'aligning {en}')
                align_transcripts(en, ARGS.closed_captions_dir, ARGS.aligned_transcripts_dir, ARGS.transcripts_dir)
            else:
                print(f'alignment already exist for {en}')
    else:
        for en in ARGS.epname:
            align_transcripts(en, ARGS.closed_captions_dir, ARGS.aligned_transcripts_dir, ARGS.transcripts_dir)
    print(f'num without scene captions: {N_WITHOUT_CAPTIONS}')
