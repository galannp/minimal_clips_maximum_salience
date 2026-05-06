from prompt_gemini import Prompter
import re
import os
import json
from dl_utils.misc import check_dir
from utils.movie_list import get_movie_list
from fuzzywuzzy import fuzz
import collections
import string

def custom_common_unique_ratio(s1, s2):
    """
    Compares strings by considering common tokens and unique tokens from each string.

    Args:
        s1: The first string.
        s2: The second string.
        scorer: The RapidFuzz scorer to use for the final comparison
                (e.g., fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio).
                fuzz.ratio is generally a good choice here.

    Returns:
        A similarity score between 0 and 100.
    """
    # 1. Tokenize and preprocess (lowercase, split)
    tokens1 = s1.lower().split()
    tokens2 = s2.lower().split()

    # Use Counter to handle potential duplicate tokens within a string
    # and to easily find intersection and differences.
    # Convert to set if you truly want unique tokens (ignoring counts within a string)
    # For this specific method, Counter makes it cleaner to represent the "full picture" of tokens
    # including their counts, even if sorted for comparison.
    counter1 = collections.Counter(tokens1)
    counter2 = collections.Counter(tokens2)

    # 2. Identify Common and Unique Tokens
    common_tokens = list((counter1 & counter2).elements()) # Elements in common, respecting counts
    unique_to_s1 = list((counter1 - counter2).elements())
    unique_to_s2 = list((counter2 - counter1).elements())

    # 3. Construct "Combined" Strings
    # Sort all components to ensure order insensitivity for the final comparison
    # This creates a "canonical" representation of the tokens in each string.
    combined_s1_tokens = common_tokens + unique_to_s1
    combined_s2_tokens = common_tokens + unique_to_s2

    # Rejoin into strings for the final fuzzy comparison
    combined_s1_str = " ".join(combined_s1_tokens)
    combined_s2_str = " ".join(combined_s2_tokens)
    
    # Handle empty strings resulting from tokenization (e.g., if input was just spaces)
    if not combined_s1_str and not combined_s2_str:
        return 100.0
    elif not combined_s1_str or not combined_s2_str:
        return 0.0

    # 4. Compare the Combined Strings
    # Using fuzz.ratio directly compares the "edit distance" between these
    # normalized representations.
    if len(combined_s1_str) <= len(combined_s2_str):
        score = fuzz.partial_ratio(combined_s1_str, combined_s2_str)
    else:
        score = fuzz.ratio(combined_s1_str, combined_s2_str)
    return score

def find_substring_line_number(text, substring):
  """
  Finds the line number of the first occurrence of a substring within a string.

  Args:
    text: The string to search within.
    substring: The substring to find.

  Returns:
    The line number (1-based index) of the first line containing the substring,
    or None if the substring is not found.
  """
  substring = re.sub(r'[^\w\s]', '', ''.join(substring.split()).lower())
  for line in text:
    line_no_space = re.sub(r'[^\w\s]', '', ''.join(line.split()).lower())
    if substring in line_no_space:
      return line
  return None

def match_substring_line_number(text, substring):
  """
  Finds the line number of the first occurrence of a substring within a string.

  Args:
    text: The string to search within.
    substring: The substring to find.

  Returns:
    The line number (1-based index) of the first line containing the substring,
    or None if the substring is not found.
  """
  substring_no_space = substring.replace('"', '').replace('-', '')
  max_line, max_score = '', 0
  for line in text:
    if 'Caption' in line and ':' in line:
        lines = [line, line.split(':', maxsplit=1)[-1]]
    else:
        lines = [line]
    for l in lines:
        line_no_space = l.replace('"', '').replace('-', '')
        new_score = custom_common_unique_ratio(substring_no_space, line_no_space)
        if new_score > max_score:
            max_score = new_score
            max_line = line
  if max_score >= 80:
    print(333333333333, substring, max_line)
    return max_line
  return None

if __name__ == '__main__':

    summary_path = 'summaries'
    #summary_path = 'summaries_2'

    key = 'ludo'
    #version = 'gemini-2.5-flash-preview-04-17'
    #version = 'gemini-1.5-flash'
    version = 'gemini-2.0-flash'

    movie_list = get_movie_list(list_id=0)
    #movie_list = ["Legion_2010", "A Walk to Remember_2002", "I'm Thinking of Ending Things_2020"]
    #movie_list = ["The Shining_1980"]

    for video_file in os.listdir('SummScreen/videos/'):
        movie_name = video_file[:-4]
        groundtruth_clips_path = f'groundtruth_clips_{version}'
        if summary_path != 'summaries':
            groundtruth_clips_path += f'_{summary_path}'
        check_dir(groundtruth_clips_path)
        if video_file[-4:] == '.mp4' and movie_name not in os.listdir(groundtruth_clips_path) and movie_name in movie_list and f'{movie_name}.json' in os.listdir(f'SummScreen/{summary_path}'):
            with open(f'SummScreen/transcripts/{movie_name}.json') as f:
                transcripts = json.load(f)['Transcript']
                transcripts_no_names = [line.split(':', maxsplit=1)[-1] for line in transcripts]
                transcripts_window = []
                transcripts_window_no_names = []
                for i in range(len(transcripts) - 3):
                    transcripts_window.append('\n'.join(transcripts[i:i + 4]))
                    transcripts_window_no_names.append('\n'.join(transcripts_no_names[i:i + 4]))
                transcripts = '\n'.join(transcripts)
                transcripts_no_names = '\n'.join(transcripts_no_names)
            with open(f'SummScreen/screenplays/{movie_name}.json') as f:
                screenplay = '\n'.join(json.load(f)['Screenplay'])
            with open(f'SummScreen/{summary_path}/{movie_name}.json') as f:
                summary = json.load(f)['soap_central']

            processed_screenplay = []
            for utt in screenplay.split('\n'):
                if utt.startswith('Caption'):
                    processed_screenplay.extend(['\n', f'{utt}', '\n'])
                else:
                    processed_screenplay.append(utt)
            screenplay = '\n'.join(processed_screenplay)

            prompt_fact_extraction = f"""Summary:
            {summary}

            For every sentence from the Summary, decompose the sentence in a list of facts (at least 1). Each fact can be only part of a sentence and should convey a single piece of information about the story.

            Example:

            Sentence 1: [Recopy the Sentence]
            *

            Sentence 2: [Recopy the Sentence]
            *

            ...

            Sentence N: [Recopy the Sentence]
            *

            """

            out = Prompter(version=version, key=key).prompt([prompt_fact_extraction], [""])[0]
            extracted_facts = []
            found_sentence = False
            for line in out.split('\n'):
                if 'Sentence' in line:
                    found_sentence = True
                if line.strip() != '' and 'Sentence' not in line and found_sentence:
                    extracted_facts.append(line)

            useful_captions = []
            visual_facts = []
            textual_facts = []
            other_facts = []

            extracted_facts_str = '\n'.join(extracted_facts)
            prompt_vis_fact_classif = f"""Screenplay:
            {screenplay}

            For every facts below:

            {extracted_facts_str}

            1. Find the information in the above Screenplay. Quote a line from the Screenplay.

            Example:

            Fact 1: Recopy the Fact
            1. Quoted line from Screenplay

            Fact 2: Recopy the Fact
            1. Quoted line from Screenplay

            ...

            Fact N: Recopy the Fact
            1. Quoted line from Screenplay

            """

            out = Prompter(version=version, key=key).prompt([prompt_vis_fact_classif], [""])[0]
            screenplay_split = screenplay.splitlines()
            iii = 0
            for line in out.split('\n'):
                if ':' in line and 'fact' in line.split(':', maxsplit=1)[0].lower():
                    curr_fact = line.split(':')[1].replace('*', '').strip()
                if line.strip()[:2] == '1.':
                    new_line = line.strip()[2:].strip()
                    found_line = find_substring_line_number(screenplay_split, new_line[3:-3])
                    if found_line is None:
                        found_line = find_substring_line_number([transcripts], new_line[3:-3])
                    if found_line is None:
                        found_line = find_substring_line_number([transcripts_no_names], new_line[3:-3])
                    if found_line is None and ':' in line:
                        new_line = new_line.split(':', maxsplit=1)[1].strip()
                        found_line = find_substring_line_number(screenplay_split, new_line[:-3])
                    if found_line is None and '(' in line:
                        new_line = '('.join(new_line.split('(')[:-1]).strip()
                        if len(new_line.split()) > 4:
                            found_line = find_substring_line_number(screenplay_split, new_line[3:])
                            print(222222222222, new_line, found_line)
                    chars = '!#&()*+,-./:;<=>?@[\]^_`{|}~'
                    if bool(re.search(f"[{re.escape(chars)}]", new_line.strip(f'{string.punctuation} '))) or len(new_line.split()) > 15:
                        if found_line is None:
                            found_line = match_substring_line_number(screenplay_split, new_line)
                        if found_line is None:
                            found_line = match_substring_line_number(transcripts_window, new_line)
                        if found_line is None:
                            found_line = match_substring_line_number(transcripts_window_no_names, new_line)
                    if found_line is None:
                        print(11111111111111111111111, line, curr_fact)
                        iii += 1
                    elif found_line.startswith('Caption'):
                        visual_facts.append(curr_fact)
                        useful_captions.append(int(found_line.split(':', maxsplit=1)[0].split('Caption')[1].strip()))
                    else:
                        textual_facts.append(curr_fact)

            print(len(extracted_facts), iii, iii / len(extracted_facts))
            useful_captions = sorted(list(set(useful_captions)))

            check_dir(groundtruth_clips_path)
            with open(f'{groundtruth_clips_path}/{movie_name}', 'w') as f:
                f.write('\n'.join(map(str, useful_captions)))

            gpt_extracted_facts_path = f'gpt-extracted-facts_{version}'
            if summary_path != 'summaries':
                gpt_extracted_facts_path += f'_{summary_path}'
            check_dir(gpt_extracted_facts_path)
            with open(f'{gpt_extracted_facts_path}/{movie_name}', 'w') as f:
                extracted_facts = [fact.replace('*', '').strip() for fact in extracted_facts]
                f.write('\n'.join(extracted_facts))
            check_dir(f'{gpt_extracted_facts_path}/vision_facts')
            with open(f'{gpt_extracted_facts_path}/vision_facts/{movie_name}', 'w') as f:
                f.write('\n'.join(visual_facts))
            check_dir(f'{gpt_extracted_facts_path}/text_facts')
            with open(f'{gpt_extracted_facts_path}/text_facts/{movie_name}', 'w') as f:
                f.write('\n'.join(textual_facts))
