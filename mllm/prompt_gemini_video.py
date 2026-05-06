import cv2
from PIL import Image
import os
import google.generativeai as genai
import json
import shutil
import time
from dl_utils.misc import check_dir
import cv2
import numpy as np

class PrompterVL():
    def __init__(self, version="gemini-1.5-pro", key='katayoun'):
        print('Initializing Gemini')
        if key == 'galann':
            # Galann's key
            genai.configure(api_key="AIzaSyCqrTgqpAPUfeRlKULCJ0_OSRcKWmWHl7k")
        elif key == 'yan':
            # Yan's key
            genai.configure(api_key="AIzaSyC2f8JKCIxWIR_ONd1XhnsnL_MrwfxMXh8")
        elif key == 'katayoun':
            # Katayoun's key
            genai.configure(api_key="AIzaSyAQhrS4eAnmhk-_cgc75IIt7klL0ePjq9M")
        elif key == 'matthias':
            # Matthias key
            genai.configure(api_key="AIzaSyC_JRRI8i2B_CIqaRSkjxJIvWMQiDQWFFE")
        elif key == 'ludo':
            genai.configure(api_key="AIzaSyA_75PUtR8ueKIr1cmQXdsUeBUbM8HFBtc")
        elif key == 'srecko':
            genai.configure(api_key="AIzaSyARLeIIZoBh4VF3eu4rNucZZ9zkpQ9BTiM")
        else:
            raise ValueError('Unrecognized Gemini API key')
        self.version = version

        # Set safety settings to allow all types of content
        self.safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }

        self.generation_config = genai.types.GenerationConfig(temperature=0.0)

        self.model = genai.GenerativeModel(model_name=self.version, tools=[])

    def prompt(self, video_path, question):
        print(f'Generating for {video_path}')

        ## USE FILE API (Not mandatory)
        print(f"Uploading file...")
        video = genai.upload_file(path=video_path)
        print(f"Completed upload: {video.uri}")

        # Check whether the file is ready to be used.
        while video.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(10)
            video = genai.get_file(video.name)

        if video.state.name == "FAILED":
            raise ValueError(video.state.name)

        max_try = 1
        ii = 0
        while ii < max_try:
            try:
                response = self.model.generate_content([question, video], safety_settings=self.safety_settings)
                if type(response) != str:
                    response = response.text
                return response
            except Exception as e:
                print(e)
                if ii == max_try:
                    print(response)
                    with open('errors', 'a') as f:
                        f.write(video_file)
                    continue
                else:
                    ii += 1
                    time.sleep(20)


if __name__ == '__main__':
    summary = '''
Okay, here's a comprehensive multimodal summary of *Forrest Gump*, based on the transcript provided:

**Opening: A Box of Chocolates and a Journey**

The movie opens with Forrest Gump, sitting at a bus stop bench, narrating his life story to strangers (the Black Woman, the White Woman). He offers a chocolate, initiating his story with his mother's famous adage: "Life was like a box of chocolates. You never know what you're gonna get.” This framing sets the tone for a life defined by unpredictable encounters and experiences.  The Black Woman’s initial complaint about hurting feet prompts him to reflect on shoes as a representation of someone's life’s journey.

**Early Life: Braces, Beliefs, and Jenny**

Forrest’s early life is characterized by physical challenges. Visually, we see Forrest with leg braces being examined by a doctor. His mother's unwavering belief in him ("Don't ever let anybody tell you they're better than you") is a constant theme and counteracts the school principal's assessment of his below-average IQ of 75. Mrs. Gump is extremely protective. She does what she must in order for Forrest to be able to attend a public school, which includes implying her husband is on vacation. She wants Forrest to have the same opportunities as the other children, despite the Principal wanting to place him in a special needs school.

Forrest's childhood home, located just outside of Greenbow, Alabama, is established. A recurring image is that of visitors passing through, staying in rented rooms. One visitor, a young Elvis Presley, picks up on Forrest's unique movements, which later influences his iconic dance. The visual of Elvis's performance being broadcasted through the store front elicits a strong and negative reaction from Mrs. Gump. The bus ride, where other children reject him, contrasts sharply with his meeting Jenny Curran. Jenny's voice becomes "the sweetest voice in the wide world," and visually, she's presented as an angelic figure. Their bond is immediate and profound. Jenny offers Forrest a seat on the school bus, marking the start of a relationship that defines Forrest’s life. He attributes his need for special shoes, "magic shoes," to a "crooked" back. Jenny asks if he is stupid, to which Forrest replies by echoing his momma’s sentiment, “Stupid is as stupid does.” The visual of Forrest and Jenny sharing activities like climbing trees, swinging, and stargazing emphasizes their deep connection. However, Jenny’s reluctance to go home hints at a troubled home life.

**Running from Bullies and a Dark Secret**

Childhood is marked by bullying, which triggers Forrest's extraordinary running ability. “Run, Forrest, run!” becomes a recurring encouragement from Jenny. This phrase visually represents his escape from physical and emotional pain. A crucial visual element is Jenny's home, a dilapidated house, representing neglect and abuse. We learn Jenny did not have a mother to protect her, as the narrator describes how her mother has "gone up to heaven." The narrative then delves into Jenny's abusive father, revealed visually through suggestive actions and dialogue ("kissing and touching"). The harrowing scene where Jenny asks Forrest to pray and become a bird so she can fly "far, far away from here" showcases both her vulnerability and the trauma she experiences.

**High School, Football, and College**

Forrest's running skills propel him to high school football stardom and a college scholarship. Visually, the scenes transition from childhood play to organized sports, showcasing Forrest's accidental achievements. The Vietnam War looms in the background.

**College Turmoil and Racial Integration**

College is depicted as a time of social upheaval. Visual cues include historical footage of Governor Wallace's stand against racial integration at the University of Alabama. Forrest inadvertently assists in the desegregation by returning a book dropped by a Black student. The dialogue underscores the racial tensions of the time, with Forrest's innocent interpretation of events ("Coons are tryin' to get into school") highlighting his naive perspective. His meeting with President Kennedy, and his need to “pee” is a moment of humor.

**Jenny's Struggles and Rejection**

Forrest visits Jenny at her college, where he interrupts her being hit by her boyfriend, Billy. When Forrest intervenes, she yells at him and tells him that he can't keep rescuing her all of the time. Jenny explains that he does not know what love is, as she recounts praying for God to turn her into a bird. Visually, Forrest offers her chocolates, then later ruins her roommate's bathrobe after falling due to being dizzy. The narrative details Jenny's dream to be a folk singer like Joan Baez and to reach people on a personal level, one-to-one. This dream is starkly contrasted by the reality of her performing at a strip club. In the end, she does not want Forrest to return to her again.

**Vietnam and Bubba**

Forrest enlists in the Army, where he meets Bubba, a fellow Alabaman obsessed with shrimp. This is presented through Bubba's detailed descriptions of various shrimp dishes, visually establishing Bubba's character as singular and driven. He promises Bubba that they will go into the shrimping business together after getting out of the Army. The transcript establishes Lt. Dan as their platoon leader, a figure of authority and military tradition, who gives them two pieces of advice for their survival: 1) take good care of your feet; 2) try not to do anything stupid, like getting yourself killed. Vietnam is depicted through the visual contrast of natural beauty ("I got to see a lot of countryside") and the harsh realities of war. The constant rain becomes a symbol of the bleakness and challenges they face.

The climactic battle scene is depicted through intense visual and auditory elements: explosions, gunfire, and the chaos of war. Forrest runs, saving several members of his platoon, including Lt. Dan, while attempting to save Bubba. Bubba's death by the river in Vietnam is a pivotal, emotional moment. He tells Forrest that he "want(s) to go home." The imagery is poignant and tragic. The visual transition from the heat of battle to the sterile environment of a military hospital highlights the contrast between war and healing.

**The Medal of Honor and Lt. Dan's Bitterness**

Forrest receives the Medal of Honor for his bravery. His meeting with President Johnson results in a humorous exchange about his buttocks wound, downplaying the gravity of the situation. The transcript details Lt. Dan's bitterness and disillusionment, who views being saved by Forrest as a betrayal of his destiny. He was supposed to die in the field, with honor. Visually, Lt. Dan is shown as angry and resentful in his hospital bed, expressing his rage at being a "cripple."

**Anti-War Protest and Reunion with Jenny**

Forrest's accidental involvement in an anti-war protest in Washington D.C. leads to a reunion with Jenny. She is introduced to his "new friends," the Black Panthers, at a rally. Her visual appearance has changed, reflecting the counterculture movement. He witnesses their differing lifestyles. Forrest then has a confrontation with Wesley, Jenny’s boyfriend, and has to stop him from hitting Jenny. He expresses his love for her and provides her with his Medal of Honor. Visually, Jenny kisses him, but leaves again.

**Ping Pong Diplomacy and National Fame**

Forrest's ping-pong skills take him to China as part of a diplomatic mission, resulting in national fame.  He meets John Lennon and talks about China and the Chinese. After returning to the USA, his fame opens doors, but he remains grounded. This is juxtaposed with the turmoil of national events, including the Watergate scandal, which Forrest unwittingly exposes. The scenes depicting President Nixon and Bob Hope are juxtaposed, showing the stark difference in how Forrest’s life juxtaposes with larger national events.

**Lieutenant Dan and Bubba Gump Shrimp**

After getting out of the Army, Forrest returns home. When he gets a check for $25,000 to say he likes a specific paddle, he returns back to Bayou La Batre to fulfil the promise he made to Bubba. Visuals of the Bubba Gump Shrimp Company forming highlight Forrest's business success. The visuals of the *Jenny* boat sailing into a hurricane underscore Lt. Dan's defiance and eventual acceptance of his fate. He yells for it to bring it on, calling it a showdown. Lt. Dan eventually makes peace with God.

**Momma's Death and Returns To Running**

Momma’s death instills in Forrest the importance of figuring out his own destiny. When her cancer returns, and he knows she is dying, he is sad and distraught. His grief prompts him to run across the country for three years, two months, fourteen days, and sixteen hours. Along the way, he amasses a following. He eventually stops his run, returns back home and moves on with his life.

**Reunion with Jenny, Fatherhood, and Loss**

A letter from Jenny brings Forrest to Savannah, where he learns he has a son, also named Forrest. This is the most important moment in his life. She is the one for whom he had spent his entire life. However, she is ill with an unnamed virus. Visual cues indicate her declining health. They marry, and he is reunited with Lieutenant Dan, who now has custom-made titanium legs and a fiancee named Susan. The joyful wedding is overshadowed by the knowledge of Jenny's impending death. Jenny dies on a Saturday morning, leaving Forrest to raise their son. Jenny is buried under their tree.

**Ending: A New Beginning**

Forrest accompanies his son on the school bus, mirroring his own childhood experience.  The story comes full circle as the cycle of life continues. The final scene shows Forrest sending Little Forrest off on the school bus, but not without his love and support. He gives his son his favorite book, "Curious George." He provides Little Forrest with a touching message, telling him that he will be right there when you get back and that he loves you.
'''
    print(PrompterVL(version='gemini-1.5-flash').prompt(video_path="/home/users/industry/cnrsatcreate/gpennec/experiments/follow_up/SummScreen/zero_shot_clips/clip_1290000.0.mp4", question=f"Summary: {summary}\n\nIs this video adding important information to the summary and should be included to the summary?"))