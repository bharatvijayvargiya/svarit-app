# filename: focus_api.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple
import spacy
import difflib
import regex as re
from collections import defaultdict

# Load your model once at startup
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

# Add CORS middleware here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

# ADD ROOT ROUTE
@app.get("/")
def root():
    return {"message": "Focus Word API is running", "endpoints": ["/get_focus_words"]}

@app.post("/get_focus_words")
def get_focus_words(data: TextRequest):
    text = data.text
    focus = extract_focus_words(text)  # Use the function from earlier
    return {"focus_words": focus}

@app.post("/minor_major_breaks")
def minor_major_breaks(data: TextRequest):
    text = data.text
    breaks = extract_minor_major_breaks(text)  # Use the function from earlier
    return {"minor_breaks": breaks['minor_breaks'], "major_breaks": breaks['major_breaks']}

def extract_focus_words(text: str):
    def map_spacy_indexes_to_split(text, doc, spacy_indexes):
        spacy_tokens = [
            t.text.strip().lower().strip(".,!?':;\"“”‘’") 
            for t in doc 
            if t.text.strip()
        ]
        split_tokens = [
            w.strip().lower().strip(".,!?':;\"“”‘’") 
            for w in text.split()
        ]

        matcher = difflib.SequenceMatcher(None, spacy_tokens, split_tokens)
        spacy_to_split_map = {}

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for offset in range(i2 - i1):
                    spacy_to_split_map[i1 + offset] = j1 + offset

        return [spacy_to_split_map[i] for i in spacy_indexes if i in spacy_to_split_map]

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # === Setup ===
    ent_token_map = {token.i: ent.label_ for ent in doc.ents for token in ent}
    CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
    SUBJECT_DEPS = {"nsubj", "nsubjpass"}
    OBJECT_DEPS = {"dobj", "pobj", "iobj"}
    CONTRASTIVE_MARKERS = {"but", "however", "yet", "although", "though", "nevertheless", "still", "whereas", "nonetheless"}
    WH_TAGS = {"WDT", "WP", "WP$", "WRB"}
    ADVERBIAL_INTENSIFIERS = {
        "very", "really", "extremely", "absolutely", "highly", "so", "too",
        "totally", "utterly", "completely", "entirely", "truly", "deeply",
        "strongly", "particularly", "especially", "significantly", "remarkably"
    }
    PENALTY_FACTOR = 0.5

    seen_concepts = set()
    seen_focus_lemmas = set()
    content_words = []
    contrast_focus_indices = set()
    wh_focus_indices = set()
    negation_focus_indices = set()
    emphatic_focus_indices = set()
    intensifier_focus_indices = set()

    for sent in doc.sents:
        main_verbs = [token for token in sent if token.dep_ == "ROOT" and token.pos_ == "VERB"]
        if not main_verbs:
            main_verbs = [token for token in sent if token.pos_ == "VERB" and token.tag_ != "MD"]
        main_verb_indices = {verb.i for verb in main_verbs}

        # Negation
        for token in sent:
            if token.dep_ == "neg":
                head = token.head
                if head.pos_ in CONTENT_POS and not head.is_stop and head.is_alpha:
                    negation_focus_indices.add(head.i)
                for child in head.children:
                    if child.dep_ in OBJECT_DEPS and child.pos_ in CONTENT_POS and not child.is_stop:
                        negation_focus_indices.add(child.i)

        # Contrast
        contrast_marker_index = next((token.i for token in sent if token.text.lower() in CONTRASTIVE_MARKERS), None)
        if contrast_marker_index is not None:
            count = 0
            for token in doc[contrast_marker_index + 1:]:
                if token.sent != sent:
                    break
                if token.pos_ in CONTENT_POS and not token.is_stop and token.is_alpha:
                    contrast_focus_indices.add(token.i)
                    count += 1
                if count >= 3:
                    break

        # WH questions
        is_question = sent.text.strip().endswith("?")
        if is_question:
            for token in sent:
                if token.tag_ in WH_TAGS:
                    wh_focus_indices.add(token.i)
                    for t in token.subtree:
                        if t.pos_ in CONTENT_POS and not t.is_stop:
                            wh_focus_indices.add(t.i)

        # Emphasis: Do-support and clefts
        for token in sent:
            if token.text.lower() in {"do", "does", "did"} and token.dep_ == "aux":
                if token.head.pos_ == "VERB" and not token.head.is_stop:
                    emphatic_focus_indices.add(token.head.i)

        for token in sent:
            if token.text.lower() == "it" and token.dep_ == "nsubj":
                copulas = [child for child in token.head.children if child.dep_ == "cop"]
                rel_clauses = [child for child in token.head.children if child.dep_ in {"relcl", "mark"}]
                if copulas and rel_clauses:
                    for child in token.head.children:
                        if child.pos_ in CONTENT_POS and not child.is_stop:
                            emphatic_focus_indices.add(child.i)
            if token.tag_ == "WDT" and token.text.lower() == "what":
                for t in token.subtree:
                    if t.pos_ in CONTENT_POS and not t.is_stop:
                        emphatic_focus_indices.add(t.i)

        # Adverbial Intensifiers
        for token in sent:
            if token.text.lower() in ADVERBIAL_INTENSIFIERS and token.pos_ == "ADV":
                for child in token.children:
                    if child.pos_ in {"ADJ", "VERB", "ADV"} and not child.is_stop:
                        intensifier_focus_indices.add(child.i)
                if token.head.pos_ in {"ADJ", "VERB", "ADV"} and not token.head.is_stop:
                    intensifier_focus_indices.add(token.head.i)

        words = [token for token in sent if token.is_alpha]
        total_words = len(words)

        for i, token in enumerate(words):
            if token.pos_ in CONTENT_POS and not token.is_stop:
                position_ratio = i / (total_words - 1) if total_words > 1 else 0
                lemma = token.lemma_.lower()
                chunk_texts = [chunk.text.lower() for chunk in token.doc.noun_chunks if token in chunk]
                concept_key = chunk_texts[0] if chunk_texts else lemma
                is_new = concept_key not in seen_concepts
                seen_concepts.add(concept_key)

                word_data = {
                    "text": token.text,
                    "index": token.i,
                    "lemma": token.lemma_,
                    "sentence": sent.text.strip(),
                    "is_main_verb": token.i in main_verb_indices,
                    "is_named_entity": token.i in ent_token_map,
                    "entity_label": ent_token_map.get(token.i, None),
                    "is_subject": token.dep_ in SUBJECT_DEPS,
                    "is_object": token.dep_ in OBJECT_DEPS,
                    "position_ratio": round(position_ratio, 2),
                    "is_sentence_initial": position_ratio <= 0.2,
                    "is_sentence_final": position_ratio >= 0.8,
                    "is_new": is_new,
                    "is_contrastive_focus": token.i in contrast_focus_indices,
                    "is_wh_focus": token.i in wh_focus_indices,
                    "is_negated": token.i in negation_focus_indices,
                    "is_emphatic": token.i in emphatic_focus_indices,
                    "is_intensified": token.i in intensifier_focus_indices,
                }

                score = 0
                score += 2.0 if word_data["is_main_verb"] else 0
                score += 1.5 if word_data["is_subject"] else 0
                score += 1.5 if word_data["is_object"] else 0
                score += 2.0 if word_data["is_named_entity"] else 0
                score += 0.5 if word_data["is_sentence_initial"] or word_data["is_sentence_final"] else 0
                score += 1.0 if word_data["is_new"] else 0
                score += 2.0 if word_data["is_contrastive_focus"] else 0
                score += 2.0 if word_data["is_wh_focus"] else 0
                score += 1.5 if word_data["is_negated"] else 0
                score += 1.5 if word_data["is_emphatic"] else 0
                score += 1.0 if word_data["is_intensified"] else 0
                if lemma in seen_focus_lemmas:
                    score *= PENALTY_FACTOR
                else:
                    seen_focus_lemmas.add(lemma)

                word_data["focus_score"] = round(score, 3)
                content_words.append(word_data)

    # Group by sentence, select top N per sentence
    sentence_map = defaultdict(list)
    for word in content_words:
        sentence_map[word["sentence"]].append(word)

    top_focus = []
    for sent, words in sentence_map.items():
        sent_len = len(sent.split())
        top_n = 1 if sent_len <= 5 else 2 if sent_len <= 12 else 3 if sent_len <= 20 else 4
        top_words = sorted(words, key=lambda w: w["focus_score"], reverse=True)[:top_n]
        top_focus.extend(top_words)

    # Remove consecutive focus words, keep only highest scoring in each cluster
    top_focus_sorted = sorted(top_focus, key=lambda w: w["index"])
    filtered_focus = []
    i = 0
    while i < len(top_focus_sorted):
        group = [top_focus_sorted[i]]
        j = i + 1
        while j < len(top_focus_sorted) and top_focus_sorted[j]["index"] == top_focus_sorted[j - 1]["index"] + 1:
            group.append(top_focus_sorted[j])
            j += 1
        best = max(group, key=lambda w: w["focus_score"])
        filtered_focus.append((best["text"], best["index"]))
        i = j

    spacy_indexes = [idx for _, idx in filtered_focus]
    split_indexes = map_spacy_indexes_to_split(text, doc, spacy_indexes)

        # Re-map focus words to use split-based indexes
    final_output = [(word, split_index) for (word, _), split_index in zip(filtered_focus, split_indexes)]
    return final_output


def extract_minor_major_breaks(text: str):
    # --- Heuristic lists ---
    PREPOSITIONS = ['in', 'on', 'at', 'by', 'for', 'from', 'to', 'with', 'about',
        'over', 'under', 'into', 'onto', 'across', 'behind', 'through',
        'between', 'among', 'after', 'before', 'during', 'against']
    CONJUNCTIONS = ['and', 'but', 'or', 'so', 'yet', 'for']
    SUBORDINATE_CONJUNCTIONS = ['because', 'although', 'if', 'when', 'since', 'while', 'unless',
        'as', 'even though', 'though', 'whereas', 'after', 'before']
    RELATIVE_PRONOUNS = ['who', 'which', 'that', 'whom', 'whose', 'which', 'that']
    ADVERBIAL_STARTERS = ['suddenly', 'eventually', 'quickly', 'slowly',
        'fortunately', 'unfortunately', 'in the morning', 'in the evening',
        'at night', 'after dinner', 'on Sunday', 'at first',
        'as a result', 'on the other hand', 'in fact', 'to be honest', 'however', 
        'meanwhile', 'in addition', 'furthermore', 'consequently', 'therefore', 
        'nevertheless', 'nonetheless', 'in conclusion', 'to summarize', 'moreover']
    TAG_QUESTIONS = ["isn't it", "don't you", "aren't they", "didn't he", "won't we",
        "right", "okay", "is it", "doesn't she", "can't you"]
    FILLERS = ["well", "oh", "um", "uh", "you know", "I mean", "so", "like", "actually"]
    VOCATIVE_TITLES = ['sir', "ma'am", 'madam', 'lord', 'lady']
    BREAK_TAGS = []

    def insert_break_after_commas_semicolons(text):
        tagged = re.sub(r'([,;])\s*', lambda m: f"{m.group(1)} <|comma|> ", text)
        BREAK_TAGS.append("comma")
        return tagged

    def insert_breaks_conjunctions(text):
        pattern = r'\s+(?=(' + '|'.join(CONJUNCTIONS) + r')\b)'
        tagged = re.sub(pattern, ' <|conjunction|> ', text)
        BREAK_TAGS.append("conjunction")
        return tagged

    def insert_breaks_prepositions(text):
        pattern = r'(?<!^)(?<![|,.])\s+(?=(' + '|'.join(PREPOSITIONS) + r')\b)'
        tagged = re.sub(pattern, ' <|preposition|> ', text)
        BREAK_TAGS.append("preposition")
        return tagged

    def insert_breaks_subordinate(text):
        pattern = r'(?<![|,.])\s+(?=(' + '|'.join(SUBORDINATE_CONJUNCTIONS) + r')\b)'
        tagged = re.sub(pattern, ' <|subordinate|> ', text)
        BREAK_TAGS.append("subordinate")
        return tagged

    def insert_breaks_relative(text):
        pattern = r'(?<![|,.])\s+(?=(' + '|'.join(RELATIVE_PRONOUNS) + r')\b)'
        tagged = re.sub(pattern, ' <|relative|> ', text)
        BREAK_TAGS.append("relative")
        return tagged

    def insert_breaks_adverbial(text):
        phrase_pattern = '|'.join(re.escape(p) for p in ADVERBIAL_STARTERS)
        tagged = re.sub(r'^(?:' + phrase_pattern + r')(?=\s)', lambda m: m.group(0) + ' <|adverbial|>', text, flags=re.IGNORECASE)
        BREAK_TAGS.append("adverbial")
        return tagged

    def insert_breaks_enumerations(text):
        tagged = re.sub(r',\s*', ' <|enumeration|> ', text)
        tagged = re.sub(r'(?<=\| [^|]+)\s+(and|or)\b', r' <|enumeration|> \1', tagged)
        BREAK_TAGS.append("enumeration")
        return tagged

    def insert_breaks_vocatives(text):
        tagged = re.sub(r'(\b[A-Z][a-z]+),', r'\1 <|vocative|>', text)
        tagged = re.sub(r', (\b[A-Z][a-z]+\b)', r'<|vocative|> \1', tagged)
        for title in VOCATIVE_TITLES:
            tagged = re.sub(r'\b(' + title + r'),', r'\1 <|vocative|>', tagged, flags=re.IGNORECASE)
            tagged = re.sub(r', (' + title + r')\b', r'<|vocative|> \1', tagged, flags=re.IGNORECASE)
        BREAK_TAGS.append("vocative")
        return tagged

    def insert_breaks_appositives(text):
        pattern = r'(\b\w+\b),\s+([^,]+?),\s+(\b\w+\b)'
        replacement = r'\1 <|appositive|> \2 <|appositive|> \3'
        prev = None
        while prev != text:
            prev = text
            text = re.sub(pattern, replacement, text)
        BREAK_TAGS.append("appositive")
        return text

    def insert_breaks_speech_patterns(text):
        tag_pattern = r'\b(' + '|'.join(re.escape(tq) for tq in TAG_QUESTIONS) + r')\b\?'
        text = re.sub(r'\s*' + tag_pattern, r' <|speech|> \1?', text, flags=re.IGNORECASE)
        filler_pattern = r'\b(' + '|'.join(re.escape(f) for f in FILLERS) + r')\b'
        text = re.sub(r'(^|\s)(' + filler_pattern + r')(\s+)', r'\1\2 <|speech|> ', text, flags=re.IGNORECASE)
        BREAK_TAGS.append("speech")
        return text

    # --- Apply all minor break tagging heuristics ---
    def insert_minor_breaks_with_tags(text):
        text = insert_break_after_commas_semicolons(text)
        text = insert_breaks_conjunctions(text)
        text = insert_breaks_prepositions(text)
        text = insert_breaks_subordinate(text)
        text = insert_breaks_relative(text)
        text = insert_breaks_adverbial(text)
        text = insert_breaks_enumerations(text)
        text = insert_breaks_vocatives(text)
        text = insert_breaks_appositives(text)
        text = insert_breaks_speech_patterns(text)
        return text

# --- Final function: get break indexes + words ---
    def get_minor_and_major_break_indexes_with_words(text):
        sentences = re.split(r'(?<=[.?!])\s+', text)
        all_minor_breaks = []
        all_major_breaks = []

        offset = 0

        for sentence in sentences:
            tagged = insert_minor_breaks_with_tags(sentence)
            tokens = tagged.split()
            original_tokens = sentence.split()
            minor_breaks = []
            major_breaks = []

            word_index = -1
            for i, token in enumerate(tokens):
                if token.startswith('<|') and token.endswith('|>'):
                    if word_index >= 0 and word_index < len(original_tokens):
                        minor_breaks.append((offset + word_index, original_tokens[word_index]))
                    continue
                word_index += 1

            for i, word in enumerate(original_tokens):
                if re.search(r'[.?!]$', word):
                    major_breaks.append((offset + i, word))

            offset += len(original_tokens)
            all_minor_breaks.append(minor_breaks)
            all_major_breaks.append(major_breaks)

        def deduplicate(breaks):
            seen = set()
            unique = []
            for sent in breaks:
                for item in sent:
                    if item[0] not in seen:
                        unique.append(item)
                        seen.add(item[0])
            return unique

        return {
            'minor_breaks': deduplicate(all_minor_breaks),
            'major_breaks': deduplicate(all_major_breaks)
        }
    return get_minor_and_major_break_indexes_with_words(text)
