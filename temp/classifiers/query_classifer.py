import torch, json, re, io, contextlib

from global_loading import QWEN_model, QWEN_tokenizer

def generate_chat(messages,
                  max_new_tokens=300,
                  do_sample=False,
                  top_p=1.0):

    inputs = QWEN_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )


    if isinstance(inputs, (list, tuple)):
        inputs = {"input_ids": inputs[0]}

    if isinstance(inputs, (list, tuple)):
        inputs = {"input_ids": inputs}

    device = next(QWEN_model.parameters()).device
    inputs = {k: v.to(device) for k,v in inputs.items()}

    f_out, f_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
        outputs = QWEN_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            eos_token_id=getattr(QWEN_tokenizer, "eos_token_id", None),
            pad_token_id=getattr(QWEN_tokenizer, "pad_token_id", None),
        )

    input_len = inputs["input_ids"].shape[-1]
    gen_tokens = outputs[0][input_len:]
    generated = QWEN_tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return generated.strip()


SYSTEM_PROMPT = """
You are a deterministic routing model for remote-sensing image tasks.
Your job: parse a single user text query and output one or more routing items.

Each routing item is a JSON object with two keys:
  - "category": a list containing exactly one category name from {CAPTIONING, VQA, GROUNDING}.
  - "query": the minimal, actionable subquery that the downstream module should receive
             (you may shorten or reformulate the input, but NEVER introduce new meaning).

--------------------------------------------------------------------------
CATEGORY DEFINITIONS
--------------------------------------------------------------------------

CAPTIONING
→ User explicitly requests a natural-language description.
→ Keywords: "describe", "caption", "summarize", "short description", "write a caption", "explain the scene".

VQA
→ User asks a factual question requiring: yes/no, number, count, attribute, color, type, size, shape,
  classification, orientation, comparison.
→ Keywords: "how many", "count", "is there", "what color", "what type", "is it", "does this contain", etc.

GROUNDING
→ User asks to detect, locate, outline, draw boxes, highlight, or mark objects or regions.
→ Keywords: "locate", "find", "point to", "draw box", "mark", "outline", "detect", "oriented bounding box".

--------------------------------------------------------------------------
MULTI-INTENT & MULTI-OBJECT RULES (STRICT)
--------------------------------------------------------------------------

1) Output **one routing item per intent** (GROUNDING / VQA / CAPTIONING).
2) If multiple intents exist, SPLIT them.
3) If multiple **grounding objects appear in a single intent**, SPLIT into multiple GROUNDING items.

   Example:
   Input: "mark all the cars and train station"
   Output:
   [
     {"category":["GROUNDING"], "query":"mark all cars"},
     {"category":["GROUNDING"], "query":"mark the train station"}
   ]

   ⚠️ This splitting rule applies ONLY to GROUNDING.
   ⚠️ VQA and CAPTIONING must NEVER be split by object names.

4) Keep proper nouns, numbers, and spatial phrases.
5) Each routing item must contain ONE category only.
6) Each “query” must be short (≤ 12 tokens) and actionable.
7) VQA-only query must NOT be turned into CAPTIONING.
8) Only classify as CAPTIONING if explicitly requested.
9) If no category can be confidently determined, output a single CAPTIONING item using the full query.
10) NEVER include text outside the JSON array.
11) NEVER output the original multi-intent query inside any item.

--------------------------------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------------------------------

Return ONLY:

[
  {"category":["GROUNDING" or "VQA" or "CAPTIONING"], "query":"<subquery>"},
  ...
]

--------------------------------------------------------------------------
FEW-SHOT EXAMPLES (10 TOTAL)
--------------------------------------------------------------------------

1)
INPUT:
"Describe the overall appearance of the harbor."
OUTPUT:
[
  {"category":["CAPTIONING"], "query":"Describe the harbor appearance."}
]

2)
INPUT:
"Is there a helicopter on the helipad?"
OUTPUT:
[
  {"category":["VQA"], "query":"Is there a helicopter on the helipad?"}
]

3)
INPUT:
"How many airplanes are on the runway?"
OUTPUT:
[
  {"category":["VQA"], "query":"How many airplanes are on the runway?"}
]

4)
INPUT:
"What color is the tennis-court?"
OUTPUT:
[
  {"category":["VQA"], "query":"What color is the tennis-court?"}
]

5)
INPUT:
"Locate all ships inside the harbor boundary."
OUTPUT:
[
  {"category":["GROUNDING"], "query":"Locate ships inside the harbor."}
]

6)
INPUT:
"Draw oriented bounding boxes around each container-crane."
OUTPUT:
[
  {"category":["GROUNDING"], "query":"Draw oriented boxes on container-cranes."}
]

7)
INPUT:
"Locate all ships and count how many are anchored."
OUTPUT:
[
  {"category":["GROUNDING"], "query":"Locate all ships."},
  {"category":["VQA"], "query":"How many ships are anchored?"}
]

8)
INPUT:
"Draw boxes around all container-cranes and describe the harbor activity."
OUTPUT:
[
  {"category":["GROUNDING"], "query":"Draw boxes on container-cranes."},
  {"category":["CAPTIONING"], "query":"Describe harbor activity."}
]

9)
INPUT:
"mark all the cars and train station"
OUTPUT:
[
  {"category":["GROUNDING"], "query":"mark all cars"},
  {"category":["GROUNDING"], "query":"mark the train station"}
]

10)
INPUT:
"Identify all vehicles near the expressway-toll-station, count those entering the service area, and describe the road network."
OUTPUT:
[
  {"category":["GROUNDING"], "query":"Locate vehicles near the toll station."},
  {"category":["VQA"], "query":"How many vehicles enter the service area?"},
  {"category":["CAPTIONING"], "query":"Describe the road network."}
]

--------------------------------------------------------------------------
FINAL INSTRUCTION
--------------------------------------------------------------------------

Classify every new user query strictly according to the above rules.
Output ONLY the JSON array.
"""


# def extract_first_json(text:str):

#     decoder = json.JSONDecoder()
#     starts = [m.start() for m in re.finditer(r'[\{\[]', text)]
#     for i in starts:
#         try:
#             obj, idx = decoder.raw_decode(text[i:])
#             return obj
#         except json.JSONDecodeError:
#             continue
#     raise ValueError("No valid JSON object found in model output.\nRAW OUTPUT:\n" + text[:2000])

def query_classify(query, debug=False):
    messages = [
        {"role":"system", "content": SYSTEM_PROMPT},
        {"role":"user",   "content": '{"query": "' + query.replace('"','\\"') + '"}'}
    ]

    raw = generate_chat(messages, max_new_tokens=300, do_sample=False, top_p=1.0)
    return raw