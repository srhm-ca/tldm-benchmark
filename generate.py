import os
import json
from tqdm.auto import tqdm
from random import shuffle, seed
import pandas
import glob
from huggingface_hub import InferenceClient
import re

ctx = []
pg_id = []
chapters = []

for l in glob.glob("books/*"):
    for pg in glob.glob(f"{l}/*"):
        ctx.append(l.split("/")[-1])
        pg_id.append(pg.split("/")[-1][2:])
        chapters.append(len(glob.glob(f"{pg}/chapter-*.txt")))


df = pandas.DataFrame(
    {
        "ctx": ctx,
        "pg_id": pg_id,
        "chapters": chapters,
    }
)

pg_metadata = pandas.read_csv("pg_metadata.csv")

model = "add model here"

api_key = "add here"

match model:
    case "model":
        llm = InferenceClient(
            base_url="url",
            api_key=api_key,
        )
        limit = ["<32k", "32k-64k", "64k-128k", "128k+"]


def prompt(m):
    return llm.chat_completion(
        [
            {
                "role": "user",
                "content": m,
            },
        ],
    )['choices'][0]['message']['content'].strip()


def user_message_prompt(m):
    return llm.chat_completion(
        m,
    )['choices'][0]['message']['content'].strip()


def annotate(text, title, author, chapter_sentence, characters=[], whole_novel=False):
    if whole_novel:
        task = "Summarize the narrative with one sentence per chapter. Describe what happens. Do not reference the narrative itself."
    else:
        task = "Summarize the narrative in a single sentence. Describe what happens. Do not reference the narrative itself."
    if type(text) is list:
        return {
            "summary": user_message_prompt(
                text + [{
                    "role": "user",
                    "content": f"""Source: {title} by {author}.

Situation: You were given a narrative. You will now be given a task about the narrative. Complete the task. Keep your response brief and to the point.

Task: {task}

{chapter_sentence}"""
                }]),
            "network": user_message_prompt(
                text + [{
                    "role": "user",
                    "content": f"""Source: {title} by {author}.

Situation: You were given a narrative. You will now be given a task about the narrative. Complete the task. Keep your response brief and to the point.

Task: List each character in the narrative and their physical location in the story.

Here are a list of possible characters in the narrative: {characters}. The list might be blank.

If the character in the above list is present in the narrative, note their last location in the narrative. If a character is in the narrative but is not in the above list, still note their last location in the narrative. Be consistent with the name. Only list characters present in the narrative. Only list individuals. Do not list groups of characters.

Please provide your response with the following JSON schema: """ + """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "Character Map",
  "description": "schema for specifying characters and locations",
  "required": [
    "individual",
    "location",
  ],
  "properties": {
    "individual": {
      "type": "string",
      "description": "First name, or description, of character in question."
    },    "location": {
      "type": "string",
      "description": "The character's last location, described in as few words as possible."
    },
  },
  "additionalProperties": false
}""" + f"""

{chapter_sentence}"""
                }]),
            "time": user_message_prompt(
                text + [{
                    "role": "user",
                    "content": f"""Source: {title} by {author}.

Situation: You were given a narrative. You will now be given a task about the narrative. Complete the task. Keep your response brief and to the point.

Task: Predict how much time in minutes, hours, or days elapsed in this narrative. Specify an exact number and unit. Only respond with a number and a unit, e.g. 30 minutes, 6 hours, or 1 day.

{chapter_sentence}"""
                }])
        }
    else:
        return {
            "summary": prompt(f"""Narrative: {text}

Source: {title} by {author}.

Situation: You were given a narrative. You will now be given a task about the narrative. Complete the task. Keep your response brief and to the point.

Task: {task}

{chapter_sentence}"""),
            "network": prompt(f"""Narrative: {text}

Source: {title} by {author}.

Situation: You were given a narrative. You will now be given a task about the narrative. Complete the task. Keep your response brief and to the point.

Task: List each character in the narrative and their physical location in the story.

Here are a list of possible characters in the narrative: {characters}. The list might be blank.

If the character in the above list is present in the narrative, note their last location in the narrative. If a character is in the narrative but is not in the above list, still note their last location in the narrative. Be consistent with the name. Only list characters present in the narrative. Only list individuals. Do not list groups of characters.

Please provide your response with the following JSON schema: """ + """{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "title": "Character Map",
  "description": "schema for specifying characters and locations",
  "required": [
    "individual",
    "location",
  ],
  "properties": {
    "individual": {
      "type": "string",
      "description": "First name, or description, of character in question."
    },
    "location": {
      "type": "string",
      "description": "The character's last location, described in as few words as possible."
    },
  },
  "additionalProperties": false
}""" + f"""

{chapter_sentence}"""),
            "time": prompt(f"""Narrative: {text}

Source: {title} by {author}.

Situation: You were given a narrative. You will now be given a task about the narrative. Complete the task. Keep your response brief and to the point.

Task: Predict how much time in minutes, hours, or days elapsed in this narrative. Specify an exact number and unit. Only respond with a number and a unit, e.g. 30 minutes, 6 hours, or 1 day.

{chapter_sentence}"""),
        }


seed(2025)
treatments = ["t1", "t2", "t3", "t4", "t5",
              "t3+t4", "t3+t5", "t4+t5", "t3+t4+t5"]
lengths = ["<32k", "32k-64k", "64k-128k", "128k+"]
windows = [0.25, 0.5, 0.75, 1]

for length in tqdm(lengths, desc="Lengths"):
    if length not in limit:
        continue
    for _, t in tqdm(df[df.ctx == length].iterrows(), total=df[df.ctx == length].shape[0], desc=f"Ground for books {length}"):
        # generate ground
        characters = []
        results = {}

        if f"{t['pg_id']}.txt" in os.listdir(f'{model}/{length}/ground/'):
            continue

        for chapter in tqdm(range(t['chapters']), desc=f"Chapters for book {t['pg_id']}"):
            title = pg_metadata[pg_metadata.pg_id ==
                                int(t["pg_id"])].iloc[0].title
            author = pg_metadata[pg_metadata.pg_id ==
                                 int(t["pg_id"])].iloc[0].author
            with open(f"books/{t["ctx"]}/pg{t["pg_id"]}/chapter-{chapter+1}.txt") as i:
                results[chapter +
                        1] = annotate(i.read(), title, author, characters)
                characters = set(list(
                    characters) + re.findall(r'"individual":\s*"([^"]+)"', results[chapter+1]["network"]))

        with open(f"{model}/{length}/ground/{t['pg_id']}.txt", 'w') as o:
            o.write(json.dumps(results))

    # generate treatments
    for _, t in tqdm(df[df.ctx == length].iterrows(), total=df[df.ctx == length].shape[0], desc=f"Treatment for books {length}"):
        for window in tqdm(windows, desc=f"Cycling window for pg{t['pg_id']}"):
            for treatment in tqdm(treatments, desc=f"{window*100}% of pg{t['pg_id']}"):
                if f"{t['pg_id']}.txt" in os.listdir(f'{model}/{length}/{window}/{treatment}/'):
                    continue

                characters = []
                results = {}
                title = pg_metadata[pg_metadata.pg_id ==
                                    int(t["pg_id"])].iloc[0].title
                author = pg_metadata[pg_metadata.pg_id ==
                                     int(t["pg_id"])].iloc[0].author

                chapters = []
                for chapter in tqdm(range(t['chapters'])):
                    with open(f"books/{t["ctx"]}/pg{t["pg_id"]}/chapter-{chapter+1}.txt") as i:
                        chapters.append(i.read())
                chapter_of_interest = f"Limit your response to the narrative from chapter 1 up until, and including, chapter {
                    int(len(chapters) * window)}."

                match treatment:
                    case "t1":  # treatment 1, no novel
                        text = ""

                    case "t2":  # treatment 2, the novel unaltered
                        text = " ".join(chapters)

                    case "t3":  # treatment 3, the novel with each chapter as a user message
                        text = [
                            {
                                "role": "user",
                                "content": c,
                            } for c in chapters
                        ]

                    case "t4":  # treatment 4, truncating the model to 25%, 50%, 75%, or 100%
                        text = " ".join(chapters[:int(len(chapters) * window)])

                    case "t5":  # treatment 5, novel, randomly suffled
                        x = chapters
                        shuffle(x)
                        text = " ".join(x)

                    case "t3+t4":  # treatment 6, 3 + 4
                        text = [
                            {
                                "role": "user",
                                "content": c,
                            } for c in chapters[:int(len(chapters) * window)]
                        ]

                    case "t3+t5":  # treatment 7, 3 + 5
                        x = chapters
                        shuffle(x)
                        text = [
                            {
                                "role": "user",
                                "content": c,
                            } for c in x
                        ]

                    case "t4+t5":  # treatment 8, 4 + 5
                        x = chapters[:int(len(chapters) * window)]
                        shuffle(x)
                        text = " ".join(x)

                    case "t3+t4+t5":  # treatment 9, 3+4+5
                        x = chapters[:int(len(chapters) * window)]
                        shuffle(x)
                        text = [
                            {
                                "role": "user",
                                "content": c,
                            } for c in x
                        ]

                results = annotate(
                    text=text,
                    title=title,
                    author=author,
                    chapter_sentence=chapter_of_interest,
                    whole_novel=True,
                )

                with open(f"{model}/{length}/{window}/{treatment}/{t['pg_id']}.txt", 'w') as o:
                    o.write(json.dumps(results))
