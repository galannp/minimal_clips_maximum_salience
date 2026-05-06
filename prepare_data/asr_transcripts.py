import whisper
import datetime
import json
import os
import argparse

def convert_seconds(seconds):
    # Create a timedelta object
    td = datetime.timedelta(seconds=seconds)
    
    # Extract hours, minutes, seconds, and milliseconds
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    milliseconds = td.microseconds // 1000
    
    # Format the output as HH:MM:SS:SSS
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_transcript_with_timestamps(video_path, output_dir):
    """
    Generates a transcript of a video with timestamps using the Whisper ASR model.

    Args:
        video_path (str): The path to the video file.
        output_file (str, optional): The name of the output text file to save the transcript.
                                     Defaults to "transcript_with_timestamps.txt".
    """
    model = whisper.load_model('small')
    model = model.to('cuda')

    result = model.transcribe(video_path, language="en")

    captions = []
    for segment in result["segments"]:
        start_time = convert_seconds(segment["start"])
        end_time = convert_seconds(segment["end"])
        text = segment["text"].strip()
        captions.append([f"{start_time} --> {end_time}", text])

    with open(f"{output_dir}/{video_path.split('/')[-1][:-4]}.json", 'w') as f:
        json.dump({'captions': captions}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epname',type=str,nargs='+',default=['all'])
    ARGS = parser.parse_args()

    for video_file_path in os.listdir('SummScreen/videos'):
        if video_file_path[-4:] == '.mp4' and f'{video_file_path[:-4]}.json' not in os.listdir(f'SummScreen/closed_captions') and (ARGS.epname == ['all'] or video_file_path[:-4] in ARGS.epname):
            generate_transcript_with_timestamps(f'SummScreen/videos/{video_file_path}', f'SummScreen/closed_captions')
