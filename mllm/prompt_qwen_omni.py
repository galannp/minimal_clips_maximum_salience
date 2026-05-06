# use conda vllm env

import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

class Qwen_Omni:
    def __init__(self, params=7):

        # default: Load the model on the available device(s)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(f"Qwen/Qwen2.5-Omni-{params}B", torch_dtype="auto", device_map="auto")
        self.model.disable_talker()

        # We recommend enabling flash_attention_2 for better acceleration and memory saving.
        #self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-Omni-7B",
        #     torch_dtype="auto",
        #     device_map="auto",
        #     attn_implementation="flash_attention_2",
        # )

        self.processor = Qwen2_5OmniProcessor.from_pretrained(f"Qwen/Qwen2.5-Omni-{params}B")

    def prompt(self, video_path, prompt):
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        # set use audio in video
        USE_AUDIO_IN_VIDEO = True

        # Preparation for inference
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        # Inference: Generation of the output text and audio
        text_ids = self.model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        response = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response.split(prompt + '\nassistant\n')[1]

if __name__ == '__main__':
    video_path = "/home/users/industry/cnrsatcreate/gpennec/experiments/follow_up/SummScreen/zero_shot_clips/The Shawshank Redemption_1994/clip_50000.mp4"
    question = "Describe what you see."
    #question = "Describe both the action and Summarize the corresponding dialogue."
    print(Qwen_Omni().prompt(video_path, question))
