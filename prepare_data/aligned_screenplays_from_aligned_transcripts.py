import os
import json
from dl_utils.misc import check_dir

for a in os.listdir("SummScreen/aligned_transcripts"):
    print(a[:-5])
    aligned_transcripts = json.load(open(f'SummScreen/aligned_transcripts/{a}'))['Transcript']
    screenplay = json.load(open(f'SummScreen/screenplays/{a}'))['Screenplay']

    aligned_screenplays = []
    i, j = 0, 0
    len_transcripts = len(aligned_transcripts)
    while i < len_transcripts or j < len(screenplay):
        if j >= len(screenplay):
            if i < len_transcripts - 1:
                print('mismatch len of transcripts and screenplay', a)
            break
        if i >= len_transcripts or aligned_transcripts[i][2] != screenplay[j] and 'Caption' in screenplay[j]:
            if i > 1:
                previous_tmstp = aligned_transcripts[i - 1][0]
            else:
                previous_tmstp = 0
            if i < len_transcripts:
                next_tmstp = aligned_transcripts[i][1]
            else:
                next_tmstp = aligned_transcripts[len_transcripts - 1][1]
            aligned_screenplays.append([previous_tmstp, next_tmstp, screenplay[j]])
            j += 1
        else:
            aligned_screenplays.append(aligned_transcripts[i])
            i += 1
            j += 1

    check_dir("SummScreen/aligned_screenplays")
    with open(f"SummScreen/aligned_screenplays/{a}", 'w') as f:
        aligned_screenplays = {'Transcript': aligned_screenplays}
        json.dump(aligned_screenplays, f, indent=4)
