# filename: focus_api.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Tuple
import spacy
from collections import defaultdict

# Load your model once at startup
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/get_focus_words")
def get_focus_words(data: TextRequest):
    text = data.text
    focus = extract_focus_words(text)  # Use the function from earlier
    return {"focus_words": focus}

def extract_focus_words(text: str):
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

    return filtered_focus
