"""Role labelling on a hand-built mini earnings call with known speakers."""

import pandas as pd

from concall.roles import (assign_speakers, build_utterances, find_qa_start, join_words, label_roles,
                           transcript_words, utterances_from_segments)


def test_join_words_reglues_numbers():
    toks = ["The", "portfolio", "grew", "by", "11", ".9", "%", "year", "-on", "-year", ",", "sir", "?"]
    assert join_words(toks) == "The portfolio grew by 11.9% year-on-year, sir?"

# (speaker, start, text) - a typical Indian concall structure
SCRIPT = [
    ("S0", 0, "Ladies and gentlemen, good day and welcome to the Q2 earnings call. I now hand the conference over to Mr. Rao."),
    ("S1", 10, "Thank you. Our net interest margin was 4.2 percent and loan growth was strong this quarter. " * 3),
    ("S2", 60, "Deposits grew 15 percent. Asset quality remained stable with GNPA at 1.3 percent. " * 3),
    ("S0", 110, "We will now begin the question-and-answer session. The first question is from Amit of XYZ Securities."),
    ("S3", 120, "Thanks for taking my question. What is the outlook for credit cost?"),
    ("S1", 130, "We expect credit cost to remain around 50 basis points."),
    ("S0", 140, "Thank you. The next question is from Priya of ABC Capital. Please go ahead."),
    ("S4", 150, "Congratulations on the quarter. Can you explain the margin compression?"),
    ("S2", 160, "Margins were impacted by deposit repricing, which should ease."),
    ("S0", 170, "That was the last question. You may now disconnect."),
]


def make_transcript_and_turns():
    segments, turns = [], []
    for i, (spk, t0, txt) in enumerate(SCRIPT):
        words, t = [], float(t0)
        for w in txt.split():
            words.append({"start": t, "end": t + 0.3, "word": w, "prob": 1.0})
            t += 0.35
        segments.append({"id": i, "start": t0, "end": t, "text": txt, "words": words})
        turns.append({"start": float(t0), "end": t, "speaker": spk})
    return {"segments": segments}, turns


def test_assign_speakers_by_overlap_and_nearest():
    words = pd.DataFrame({"start": [0.0, 5.0, 9.6], "end": [0.5, 5.5, 9.9], "word": ["a", "b", "c"]})
    turns = [{"start": 0, "end": 1, "speaker": "A"}, {"start": 5.2, "end": 9, "speaker": "B"}]
    assert assign_speakers(words, turns).tolist() == ["A", "B", "B"]  # c is 0.75 s after B's turn: nearest


def test_roles_from_diarized_call():
    tr, turns = make_transcript_and_turns()
    words = transcript_words(tr)
    words["speaker"] = assign_speakers(words, turns)
    utts = build_utterances(words)
    qa = find_qa_start(utts)
    assert qa == 110
    labelled, evidence = label_roles(utts, qa)
    role = labelled.groupby("speaker")["role"].first().to_dict()
    assert role == {"S0": "moderator", "S1": "management", "S2": "management",
                    "S3": "analyst", "S4": "analyst"}
    assert set(labelled.loc[labelled["start"] < 110, "section"]) == {"prepared"}
    assert evidence["moderator"] == "S0"


def test_text_fallback_without_diarization():
    tr, _ = make_transcript_and_turns()
    utts = utterances_from_segments(tr)
    labelled, ev = label_roles(utts, find_qa_start(utts))
    assert ev["method"] == "text"
    roles = labelled["role"].tolist()
    assert roles[0] == "moderator" and roles[1] == "management"
    assert roles[4] == "analyst" and roles[5] == "management"
