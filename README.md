# [Minimal Clips, Maximum Salience: Long Video Summarization via Key Moment Extraction (IWSDS 2026)](https://aclanthology.org/2026.iwsds-1.22/)

[[📝 Paper]](https://aclanthology.org/2026.iwsds-1.22/) [[📚 Bibtex]](#citation)

_Galann Pennec, Zhengyuan Liu, Nicholas Asher, Philippe Muller, Nancy Chen_

<p align="center">
<img src="./figs/pipeline.png" width="768"/>
</p>

Here we provide code for our Video Clip Selection pipeline for Efficient Long Movie Summarization.

Understanding pipeline agentic framework for very long video understanding powered by entity scene graphs, EGAgent. EGAgent consists of a planning agent equipped for multi-hop cross-modal reasoning by querying three tools: a visual search tool, an audio transcript search tool, and an entity graph search tool. We structure this repository as follows:
1. [Preparing the Data](#preparing-the-data)
2. [Cutting the Video in 20 seconds video clips](#cutting-the-video-in-20-seconds-video-clips)
3. [Video Clip Selection](#video-clip-selection) using one of the implemented approaches ([Our Clip Selection](#our-clip-selection), [Random Clips](#baseline-1-random-clips) or [Silent Clips](#baseline-2-silent-clips))
4. [Screenplay Summarization](#screenplay-summarization)
5. [Evaluation](#evaluation)

## Installation
Install prerequisite packages with conda.
```
conda create -n minimal_clips python=3.12
conda activate minimal_clips
pip install -r requirements.txt
```

## Configure Paths
Update dataset, model, and API key locations in [`paths.py`](paths.py) before running the scripts.

## Load the Data
1. Download the videos
Our experiments are made on the test split of the [MovieSum](https://huggingface.co/datasets/rohitsaxena/MovieSum) dataset made of 200 movies for evaluation.
Download the videos in .mp4 for all the movies on [MovieBox](http://moviebox.ph/)

2. Extract the annotations in MovieSum test split
```
python -m load_data.extract_annotated_data
```

## Prepare the Data
1. Aligning Videos and Dialogues in Time
```
python -m prepare_data.asr_transcripts
python -m prepare_data.align_vid_and_transcripts --closed_captions_dir SummScreen/closed_captions --aligned_transcripts_dir SummScreen/aligned_screenplays --transcripts_dir SummScreen/screenplays
python -m prepare_data.aligned_screenplays_from_aligned_transcripts
```

2. Cutting the Videos in 20 seconds video clips
```
python -m prepare_data.video_segmentation
```

## Video Clip Selection
We set the number of selected clips `--nb_clips` to 50 in what follows (other values studied in the paper: 25, 75).
Choose one of the following Clip Selection approaches:

### Our Clip Selection
1. Lightweight Captioning of all Video Clips
```
python -m clip_selection.ours.lightweight_captioning
```

2. Clip Selection
Flags:
- `--few_shot_examples`: Indicates that few shot examples are being used by the clip selection method.
```
python -m clip_selection.ours.clip_selection --nb_clips 50 --few_shot_examples --api_key `YOUR_GOOGLE_API_KEY`
```

3. [Optional] Recaptioning of the Selected Clips
```
python -m summ.recaption_clips --clip_selection ours --nb_clips 50 --api_key `YOUR_GOOGLE_API_KEY`
```

### Baseline 1: Random Clips
1. Random Clips Selection
```
python -m clip_selection.random.random_clip_selection --nb_clips 50
```

2. [Optional] Recaptioning of the Selected Clips
```
python -m summ.recaption_clips --clip_selection random --nb_clips 50 --api_key `YOUR_GOOGLE_API_KEY`
```

### Baseline 2: Silent Clips
1. Silent Clips Selection
```
python -m clip_selection.silent.silent_clip_selection --nb_clips 50
```

2. [Optional] Recaptioning of the Selected Clips
Flags:
- `--captioning_model`: the models used for recaptioning the silent clips (use `gemini-2.5-flash-lite` or `qwen_omni`)
```
python -m clip_selection.silent.extract_silent_clips --nb_clips 50
python -m clip_selection.silent.recaption_silent_clips --captioning_model gemini-2.5-flash-lite --api_key `YOUR_GOOGLE_API_KEY`
```

### Clip Selection Reference
1. Clip Selection Reference
```
python -m upperbound.groundtruth_visual_facts
```

2. [Optional] Recaptioning of the Selected Clips
```
python -m upperbound.extract_groundtruth_clips.py
python -m summ.recaption_clips --clip_selection upperbound --api_key `YOUR_GOOGLE_API_KEY`
```

## Clip Selection Evaluation
2. Clip Selection Evaluation against the Reference
By running the command below we are able to generate the following plots displaying the Recall at $K$ for varying $\tau$, where $K$ is the number of selected clips and $\tau$ is the threshold for the matching of a retrieved clip to the reference.
Flags:
- `--K`: the number of selected clips by the tested methods for the plot
```
python -m clip_selection.eval.plot_recall_at_k --K 50
```
<p align="center">
<img src="./figs/plot_25.png" width="256"/>
</p>

<p align="center">
<img src="./figs/plot_50.png" width="256"/>
</p>

<p align="center">
<img src="./figs/plot_75.png" width="256"/>
</p>


## Screenplay Summarization
1. Build the Screenplays
Flags:
- `--clip_selection`: the clip selection algorithm used. Choose between `ours`, `random` or `silent`.
- `--nb_clips`: number $K$ of selected clips by the method. Studied values in the paper: 25, 50 and 75.
Optional flags:
- `--recaptioning`: number $K$ of selected clips by the method. Studied values in the paper: 25, 50 and 75.
- `--few_shot_examples`: Indicates that few shot examples are being used by the clip selection method.
```
python -m summ.build_generated_screenplay --clip_selection ours --nb_clips 50 --recaptioning --few_shot_examples
```

2. Summarize the Screenplays
Flags:
- `--path`: directory path to the built screenplays in the previous step
- `--version`: summarization model. Choose between `gemini-2.5-flash`, `gemini-1.5-flash` or `qwen2-72b-instruct`
- `--api_key`: the `GOOGLE_API_KEY`
- `--nb_words`: in all our experiments we set the summary length to 1000 words.
```
python -m summ.summarize_screenplay --version gemini-2.5-flash --api_key `YOUR_GOOGLE_API_KEY` --path `built_screenplays_dir` --nb_words 1000
```

## [Optional] Screenplay of Reference Clips
1. Build the Screenplay of Reference Clips
```
python -m summ.upperbound.screenplay_upperbound_clips
```

2. Summarize the Screenplay of Reference Clips
```
python -m summ.summarize_screenplay --version gemini-2.5-flash --api_key `YOUR_GOOGLE_API_KEY` --path `reference_screenplays_dir` --nb_words 1000
```

## Evaluation
1. Evaluation on Traditional Metrics (ROUGE and METEOR)
Flags:
- `--expname`: directory path to the generated summaries.
- `--truncate_length`: in our metrics computations, we always truncate the summaries to 1000 words.
```
python -m summ.eval.traditional_metrics --truncate_length 1000 --expname `summmary_dir`
```

2. Evaluation on Multimodal Metrics ([MFactSum](https://aclanthology.org/2025.ijcnlp-long.129/))
```
python -m summ.eval.MFactSum_fact_eval --truncate_length 1000 --expname `summmary_dir` --api_key `YOUR_GOOGLE_API_KEY`
```

## Citation
If you find this project useful in your research, please consider citing:
```
@misc{pennec2026minimalclipsmaximumsalience,
      title={Minimal Clips, Maximum Salience: Long Video Summarization via Key Moment Extraction}, 
      author={Galann Pennec and Zhengyuan Liu and Nicholas Asher and Philippe Muller and Nancy F. Chen},
      year={2026},
      eprint={2512.11399},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2512.11399}, 
}
```

## License
This code is CC-BY-NC4.0 licensed, as found in the [LICENSE](LICENSE.md) file.
