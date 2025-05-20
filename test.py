import re
import ast
import json
import pickle
import glob
import numpy as np
import pandas
from math import log, exp
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
model = SentenceTransformer("all-MiniLM-L6-v2")

# load dataset
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

# declare indepedent variables & models
treatments = ["t1", "t2", "t3", "t4", "t5", "t3+t4", "t3+t5", "t4+t5", "t3+t4+t5"]
lengths = ["<32k", "32k-64k", "64k-128k", "128k+"]
windows = [0.25, 0.5, 0.75, 1]
models = [
    ("model", ["<32k", "32k-64k", "64k-128k", "128k+"]),
]

# helper functions
def extract_network(text, cluster=True):
    char_locs = {}

    for pattern in [
        r'["\']{0,1}individual["\']{0,1}\s*:\s*["\']{1}([^"\']+)["\']{1}\s*,\s*["\']{0,1}location["\']{0,1}\s*:\s*["\']{1}([^"\']+)["\']{1}',
        r'["\']{1}([^"\']+)["\']{1}\s*:\s*["\']{1}([^"\']+)["\']{1}'
    ]:
        for match in re.finditer(pattern, text):
            char, loc = match.group(1).strip(), match.group(2).strip()

            if ("not present" in loc.lower() or 
                (pattern == r'["\']{1}([^"\']+)["\']{1}\s*:\s*["\']{1}([^"\']+)["\']{1}' and
                 (any(x in char.lower() for x in ["individual", "location", "summary", "time", "network"]) or
                  any(x in loc for x in [":", "{", "}"]) or
                  len(char) == 0 or len(loc) == 0))):
                continue
                
            char_locs[char.lower()] = (char, loc)
        
        if char_locs:  # Stop if first pattern found matches
            break
    
    if not cluster or not char_locs:
        return list(char_locs.values())
    
    try:
        all_chars = [pair[0] for pair in char_locs.values()]
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": 
                      f"The following list contains character names from a novel. Group similar names that refer to the same character on single lines separated by pipes (|). Include each name exactly once. DO NOT combine family members or different genders. Start each line with an asterisk.\n\n{all_chars}"}]
        ).choices[0].message.content

        char_map = {}
        for cluster in [r.replace("*", "").strip().split("|") for r in response.split("\n")]:
            if cluster and any(cluster):
                canonical = cluster[0].strip()
                for char in cluster:
                    char_map[char.strip().lower()] = canonical

        return [(char_map.get(k, v[0]), v[1]) for k, v in char_locs.items()]
    except:
        return list(char_locs.values())


def network_similarity(list1, list2):
    if not list1 or not list2:
        return 0.0
    
    dict1 = {char.lower(): (char, loc) for char, loc in list1}
    dict2 = {char.lower(): (char, loc) for char, loc in list2}
    
    exact_matches = set(dict1.keys()).intersection(set(dict2.keys()))
    remaining1 = set(dict1.keys()) - exact_matches
    remaining2 = list(set(dict2.keys()) - exact_matches)
    
    all_matches = []
    
    for char in exact_matches:
        try:
            embed1 = model.encode(dict1[char][1])
            embed2 = model.encode(dict2[char][1])
            sim = model.similarity(embed1, embed2)
            sim_value = float(sim[0][0] if hasattr(sim, 'shape') and len(sim.shape) > 1 else sim)
            all_matches.append((char, sim_value))
        except:
            all_matches.append((char, 0.1))
    
    for char1 in remaining1:
        if not remaining2:
            break
        try:
            char_embeds = model.encode(remaining2)
            char_sims = model.similarity(model.encode(char1), char_embeds)
            char_sims = char_sims.detach().numpy() if hasattr(char_sims, 'detach') else char_sims
            
            best_idx = np.argmax(char_sims)
            best_sim = float(char_sims[best_idx])  # Ensure it's a float
            
            if best_sim >= 0.6:
                matched_char = remaining2[best_idx]
                loc_sim = model.similarity(
                    model.encode(dict1[char1][1]), 
                    model.encode(dict2[matched_char][1])
                )
                loc_sim_value = float(loc_sim[0][0] if hasattr(loc_sim, 'shape') and len(loc_sim.shape) > 1 else loc_sim)
                all_matches.append((char1, best_sim * loc_sim_value))
                remaining2.remove(matched_char)
        except:
            continue
    
    intersection = sum(weight for _, weight in all_matches)
    union = len(dict1) + len(dict2) - len(all_matches)
    
    result = intersection / union if union > 0 else 0.0
    return float(result)


def time_similarity(observed_time, ground_time):
    if ground_time == 0:
        return 1.0 if observed_time == 0 else 0.0
    print(observed_time)
    print(ground_time)
    rel_error = abs(observed_time - ground_time) / ground_time
    similarity = 1 / (1 + rel_error)
    return similarity


def summary_similarity(ground_embedding, observed_embedding):
    similarity = model.similarity(ground_embedding, observed_embedding)
    if hasattr(similarity, 'tolist'):
        sim_value = similarity.tolist()[0]
        if isinstance(sim_value, list):
            sim_value = sim_value[0]
    else:
        sim_value = float(similarity)
    return max(0.0, min(1.0, sim_value))


def render_time(s, return_t=False):
    m = re.findall(r'(\d+)\s+(minute|minutes|second|seconds|day|days|month|months|year|years|hour|hours)', s)
    c = {
        'second': 1,
        'seconds':1,
        'minute': 60,
        'minutes': 60,
        'day': 86400,
        'days': 86400,
        'month': 2592000,
        'months': 2592000,
        'year': 31536000,
        'years': 31536000,
        'hour': 3600,
        'hours': 3600
    }
    t = 0
    for n, u in m: 
        t += int(n) * c[u] 
    if return_t:
        return t if t > 0 else 0
    elif t < 60: 
        return f"{t} second{'s'if t!=1 else''}"
    elif t < 3600:
        m = t // 60
        return f"{m} minute{'s'if m!=1 else''}"
    elif t < 86400:
        h = t // 3600
        return f"{h} hour{'s'if h!=1 else''}"
    elif t < 2592000:
        d = t // 86400
        return f"{d} day{'s'if d!=1 else''}"
    elif t < 31536000:
        m = t // 2592000
        return f"{m} month{'s'if m!=1 else''}"
    else:
        y = t // 31536000
        return f"{y} year{'s'if y!=1 else''}"
    

# generate ground & observed values

client = AzureOpenAI(api_key="insert api key", api_version='2025-03-01-preview', azure_endpoint="insert endpoint for gpt-4.1")

ground = {}

for m in models:
    if m[0] not in ground.keys():
        ground[m[0]] = {}
    for l in m[1]:
        if l not in ground[m[0]].keys():
            ground[m[0]][l] = {}
        for w in windows:
            if w not in ground[m[0]][l].keys():
                ground[m[0]][l][w] = {}
            for _, b in df[df.ctx == l].iterrows():
                if b['pg_id'] not in ground[m[0]][l][w].keys():
                    ground[m[0]][l][w][b['pg_id']] = {}
                with open(f'{m[0]}/{l}/ground/{b['pg_id']}.txt') as i:
                    d = json.loads(i.read())
                    d = [d[str(c+1)] for c in range(b['chapters'])]
                    d = d[:int(len(d) * w)]
                    # summary
                    if 'summary' not in ground[m[0]][l][w][b['pg_id']].keys():
                        ground[m[0]][l][w][b['pg_id']]['summary'] = model.encode(
                            " ".join(
                                [
                                    c["summary"] for c in d if c["summary"]
                                ]
                            )
                        )
                    # time
                    if 'time' not in ground[m[0]][l][w][b['pg_id']].keys():
                        ground[m[0]][l][w][b['pg_id']]['time'] = (
                            model.encode(render_time(str(d))),
                            render_time(str(d), return_t=True)
                        )
                    # network
                    if 'network' not in ground[m[0]][l][w][b['pg_id']].keys():
                        ground[m[0]][l][w][b['pg_id']]['network'] = extract_network(
                            bytes(
                                str(d),
                                'utf-8'
                            ).decode('unicode_escape')
                        )

observed = {}

for m in models:
    if m[0] not in observed.keys():
        observed[m[0]] = {}
    for l in m[1]:
        if l not in observed[m[0]].keys():
            observed[m[0]][l] = {}
        for w in windows:
            if w not in observed[m[0]][l].keys():
                observed[m[0]][l][w] = {}
            for t in treatments:
                if t not in observed[m[0]][l][w].keys():
                    observed[m[0]][l][w][t] = {}
                for _, b in df[df.ctx == l].iterrows():
                    if b['pg_id'] not in observed[m[0]][l][w][t].keys():
                        observed[m[0]][l][w][t][b['pg_id']] = {}
                    with open(f'{m[0]}/{l}/{w}/{t}/{b['pg_id']}.txt') as i:
                        text = i.read()
                        d = json.loads(text)
                        if 'summary' not in observed[m[0]][l][w][t][b['pg_id']].keys():
                            observed[m[0]][l][w][t][b['pg_id']]['summary'] = model.encode(d["summary"] if d["summary"]  else "")
                        if 'time' not in observed[m[0]][l][w][t][b['pg_id']].keys():
                            observed[m[0]][l][w][t][b['pg_id']]['time'] = (
                                model.encode(d["time"] if d["time"]  else ""),
                                render_time(d["time"] if d["time"] else "", return_t=True)
                            )
                        if 'network' not in observed[m[0]][l][w][t][b['pg_id']].keys():
                            observed[m[0]][l][w][t][b['pg_id']]['network'] = extract_network(
                                bytes(
                                    text,
                                    'utf-8'
                                ).decode('unicode_escape')
                            )

# save data 
with open('observed.pkl', 'wb') as o:
    pickle.dump(observed, o)

with open('ground.pkl', 'wb') as o:
    pickle.dump(ground, o)

# generate results

with open('ground.pkl', 'rb') as i:
    ground = pickle.load(i)

with open('observed.pkl', 'rb') as i:
    observed = pickle.load(i)

results = {}

for m in models:
    if m[0] not in results:
        results[m[0]] = {}
    for l in m[1]:
        if l not in results[m[0]]:
            results[m[0]][l] = {}
        for w in windows:
            if w not in results[m[0]][l]:
                results[m[0]][l][w] = {}
            for t in treatments:
                if t not in results[m[0]][l][w]:
                    results[m[0]][l][w][t] = {}
                summary = []
                time = []
                network = []
                for _, b in df[df.ctx == l].iterrows():
                    # summary
                    summary.append(summary_similarity(
                        ground[m[0]][l][w][b['pg_id']]['summary'],
                        observed[m[0]][l][w][t][b['pg_id']]['summary']
                    ))
                    # time
                    time.append(time_similarity(
                        observed[m[0]][l][w][t][b['pg_id']]['time'][1],
                        ground[m[0]][l][w][b['pg_id']]['time'][1]
                    ))
                    # network
                    network.append(network_similarity(
                        ground[m[0]][l][w][b['pg_id']]['network'],
                        observed[m[0]][l][w][t][b['pg_id']]['network']
                    ))
                summary_avg = sum(summary)/len(summary) if summary else 0
                time_avg = sum(time)/len(time) if time else 0
                network_avg = sum(network)/len(network) if network else 0
                overall_avg = (summary_avg + time_avg + network_avg)/3
                results[m[0]][l][w][t] = {
                    'summary': summary_avg,
                    'time': time_avg,
                    'network': network_avg,
                    'average': overall_avg,
                    'raw': {
                        'summary': summary,
                        'time': time,
                        'network': network
                    }
                }

# calculate results

for m in models:
    for l in m[1]:
        for w in ["0.25", "0.5", "0.75", "1"]:
            for t in treatments:
                time = []
                for _, b in df[df.ctx == l].iterrows():
                    time.append(time_similarity(
                        observed[m[0]][l][float(w)][t][b['pg_id']]['time'][1],
                        ground[m[0]][l][float(w)][b['pg_id']]['time'][1]
                    ))
                print(results[m[0]][l][w][t])
                time_avg = sum(time)/len(time) if time else 0
                results[m[0]][l][w][t]['time'] = time_avg
                results[m[0]][l][w][t]['raw']['time'] = time
                results[m[0]][l][w][t]['average'] = (results[m[0]][l][w][t]['summary'] + time_avg + results[m[0]][l][w][t]['network'])/3