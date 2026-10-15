"""Materialize the authored benchmark; never replace an existing dataset implicitly."""

import argparse
import json
from pathlib import Path

from evaluation.dataset import ROOT, json_sha256


# Questions and references were authored against the checked-in corpus, not collected from users.
# Multiple questions reuse each record. This is a development/regression benchmark.
ATTRACTIONS = {
    "religious_1": [
        ("How tall is the Hanuman statue at Hanuman Vatika?", "The Hanuman statue at Hanuman Vatika is 75 feet tall."),
        ("Which Rourkela temple complex features a 75-foot Hanuman statue?", "Hanuman Vatika features the 75-foot Hanuman statue."),
        ("What facilities besides the main statue are described at Hanuman Vatika?", "Hanuman Vatika contains multiple shrines, gardens, and a kalyan mandap."),
        ("When should I visit Hanuman Vatika for cooler weather?", "Visit Hanuman Vatika in the evening for cooler weather."),
        ("Which attraction has a kalyan mandap for community gatherings?", "Hanuman Vatika has a kalyan mandap and is suitable for community gatherings."),
        ("Can a family visit Hanuman Vatika together?", "Hanuman Vatika is suitable for families and community gatherings."),
    ],
    "religious_2": [
        ("To whom is Vedvyas Temple dedicated?", "Vedvyas Temple is dedicated to sage Vedvyas, the author of the Mahabharata."),
        ("Which rivers meet near Vedvyas Temple?", "The Brahmani, Koel, and Saraswati rivers meet near Vedvyas Temple."),
        ("How far from Rourkela is Vedvyas Temple according to the record?", "Vedvyas Temple is approximately 20 km from Rourkela."),
        ("Where can I visit a shrine associated with the author of the Mahabharata?", "Vedvyas Temple is dedicated to sage Vedvyas, the author of the Mahabharata."),
        ("Which temple is situated at a three-river confluence?", "Vedvyas Temple is at the confluence of the Brahmani, Koel, and Saraswati rivers."),
        ("How does the record describe the atmosphere at Vedvyas Temple?", "Vedvyas Temple is peaceful and spiritual, and usually less crowded."),
    ],
    "nature_1": [
        ("What are the opening hours of Indira Gandhi Park in the dataset?", "Indira Gandhi Park is open from 8 AM to 6 PM."),
        ("What is the recorded entry fee at Indira Gandhi Park?", "The recorded entry fee at Indira Gandhi Park is Rs 10."),
        ("What attractions are inside Indira Gandhi Park?", "Indira Gandhi Park has a small zoo, a toy train, and a musical fountain."),
        ("Which public park offers a toy train and a small zoo?", "Indira Gandhi Park offers a toy train and a small zoo."),
        ("When is a cooler time for a family visit to Indira Gandhi Park?", "The cool morning hours are suitable for a family visit to Indira Gandhi Park."),
        ("Where can children see a musical fountain along with a toy train?", "Indira Gandhi Park has a musical fountain and a toy train."),
    ],
    "religious_7": [
        ("Where is the Jagannath Temple located in Rourkela?", "The Jagannath Temple is in Sector 3 of Rourkela."),
        ("Which famous temple is the Sector 3 Jagannath Temple a replica of?", "It is a replica of the Puri Jagannath Temple."),
        ("What annual festival does the Jagannath Temple in Sector 3 host?", "The Jagannath Temple in Sector 3 hosts the annual Rath Yatra."),
        ("Which Rourkela shrine is modeled after the temple at Puri?", "The Jagannath Temple in Sector 3 is a replica of the Puri Jagannath Temple."),
        ("Where in the city can I find a temple associated with Rath Yatra?", "The Jagannath Temple in Sector 3 hosts Rath Yatra."),
        ("How is the maintenance of Jagannath Temple Sector 3 described?", "The Jagannath Temple in Sector 3 is described as well-maintained."),
    ],
    "nature_3": [
        ("Why was Mandira Dam constructed?", "Mandira Dam was constructed to supply water to Rourkela Steel Plant."),
        ("Where is Mandira Dam located according to the description?", "Mandira Dam is near Kansbahal's foothills."),
        ("What leisure activity is available at Mandira Dam?", "Mandira Dam is a picnic spot with boating facilities."),
        ("Which reservoir supplies water to the steel plant?", "Mandira Dam supplies water to Rourkela Steel Plant."),
        ("When is a good season to visit Mandira Dam?", "Visit Mandira Dam during cool weather, especially post-monsoon."),
        ("Where can I picnic and go boating near Kansbahal?", "Mandira Dam is a picnic spot with boating facilities near Kansbahal's foothills."),
    ],
    "nature_4": [
        ("How far is Pitamahal Dam from Rourkela in the dataset?", "Pitamahal Dam is about 22 km from Rourkela."),
        ("What surrounds Pitamahal Dam?", "Pitamahal Dam is surrounded by hills and forests."),
        ("What activities are mentioned for Pitamahal Dam?", "Visitors enjoy boating, scenic photography, and picnics at Pitamahal Dam."),
        ("Which dam about 22 kilometres away offers scenic photography?", "Pitamahal Dam is about 22 km from Rourkela and offers scenic photography."),
        ("When is the best time to visit Pitamahal Dam?", "Winter and post-monsoon are the best times to visit Pitamahal Dam."),
        ("Where is there a reservoir surrounded by hills and forests for a picnic?", "Pitamahal Dam is a reservoir surrounded by hills and forests, suitable for picnics."),
    ],
    "nature_11": [
        ("What activities are suitable at Koel Riverbank?", "Koel Riverbank is suitable for relaxation, picnics, and evening walks."),
        ("What scenery does Koel Riverbank offer?", "Koel Riverbank offers views of the flowing river and surrounding greenery."),
        ("How is Koel Riverbank described?", "Koel Riverbank is a peaceful riverside spot near Rourkela."),
        ("Where can I take an evening walk beside the Koel river?", "Koel Riverbank is suitable for evening walks."),
        ("Which riverside place near Rourkela is described as peaceful for relaxation?", "Koel Riverbank is a peaceful riverside spot suitable for relaxation."),
        ("Can I picnic at Koel Riverbank?", "Koel Riverbank is suitable for picnics."),
    ],
    "religious_9": [
        ("Which goddess is Rani Sati Mandir dedicated to?", "Rani Sati Mandir is dedicated to Goddess Rani Sati."),
        ("When does Rani Sati Mandir attract devotees?", "Rani Sati Mandir attracts devotees during daily prayers and festivals."),
        ("Is Rani Sati Mandir described as being within city limits?", "Rani Sati Mandir is described as located within city limits."),
        ("Which easily accessible spiritual destination honors Goddess Rani Sati?", "Rani Sati Mandir honors Goddess Rani Sati and is easily accessible."),
        ("What regular worship activity is described at Rani Sati Mandir?", "Devotees attend daily prayers at Rani Sati Mandir."),
        ("How accessible is Rani Sati Mandir according to the record?", "Rani Sati Mandir is within city limits and easily accessible."),
    ],
    "nature_5": [
        ("On which river is Darjeeng Picnic Spot located?", "Darjeeng Picnic Spot is on the Brahmani River."),
        ("How far from Rourkela is Darjeeng Picnic Spot?", "Darjeeng Picnic Spot is about 45 km from Rourkela."),
        ("What natural surroundings are described at Darjeeng?", "Darjeeng is surrounded by hills and dense forests."),
        ("Which picnic spot is on the Brahmani River about 45 kilometres away?", "Darjeeng Picnic Spot is on the Brahmani River, about 45 km from Rourkela."),
        ("When is a good time to visit Darjeeng Picnic Spot?", "Winter and early summer are good times to visit Darjeeng Picnic Spot."),
        ("Where can I enjoy riverside photography among hills and dense forests?", "Darjeeng Picnic Spot offers riverside photography among hills and dense forests."),
    ],
    "nature_13": [
        ("Where is Mirigikhoj Waterfall described as being?", "Mirigikhoj Waterfall is a hidden natural waterfall near Rourkela."),
        ("What surrounds Mirigikhoj Waterfall?", "Mirigikhoj Waterfall is surrounded by forest."),
        ("When is the best time to visit Mirigikhoj Waterfall?", "Post-monsoon and winter are the best times to visit Mirigikhoj Waterfall."),
        ("Which hidden waterfall near Rourkela is suitable for trekking?", "Mirigikhoj Waterfall is suitable for trekking."),
        ("What activities are recommended at Mirigikhoj Waterfall?", "Trekking and nature photography are suitable at Mirigikhoj Waterfall."),
        ("Does the dataset provide latitude and longitude for Mirigikhoj Waterfall?", "No. The latitude and longitude for Mirigikhoj Waterfall are null in the dataset."),
    ],
}

MULTI = [
    (["religious_1", "nature_1"], "Compare the family facilities at Hanuman Vatika and Indira Gandhi Park.",
     "Hanuman Vatika has shrines, gardens, and a kalyan mandap. Indira Gandhi Park has a small zoo, toy train, and musical fountain."),
    (["nature_3", "nature_4"], "Which seasons do the records recommend for Mandira Dam and Pitamahal Dam?",
     "Mandira Dam is best during cool weather, especially post-monsoon. Pitamahal Dam is best in winter and post-monsoon."),
    (["religious_2", "nature_5"], "Which rivers are associated with Vedvyas Temple and Darjeeng Picnic Spot?",
     "Vedvyas Temple is at the confluence of the Brahmani, Koel, and Saraswati rivers. Darjeeng Picnic Spot is on the Brahmani River."),
    (["religious_7", "religious_9"], "Describe the religious associations of Jagannath Temple Sector 3 and Rani Sati Mandir.",
     "Jagannath Temple Sector 3 is a replica of the Puri Jagannath Temple and hosts Rath Yatra. Rani Sati Mandir is dedicated to Goddess Rani Sati."),
    (["nature_11", "nature_5"], "What can visitors do at Koel Riverbank and Darjeeng Picnic Spot?",
     "Koel Riverbank offers relaxation, picnics, and evening walks. Darjeeng Picnic Spot offers picnics and photography."),
    (["nature_4", "nature_13"], "Compare the natural surroundings of Pitamahal Dam and Mirigikhoj Waterfall.",
     "Pitamahal Dam is surrounded by hills and forests. Mirigikhoj Waterfall is surrounded by forest."),
    (["religious_1", "religious_2"], "What are the main religious features of Hanuman Vatika and Vedvyas Temple?",
     "Hanuman Vatika has a 75-foot Hanuman statue. Vedvyas Temple is dedicated to sage Vedvyas, author of the Mahabharata."),
    (["nature_3", "nature_5"], "Identify the main setting of Mandira Dam and Darjeeng Picnic Spot.",
     "Mandira Dam is a reservoir near Kansbahal's foothills. Darjeeng Picnic Spot is on the banks of the Brahmani River."),
    (["nature_1", "nature_13"], "Which activities distinguish Indira Gandhi Park from Mirigikhoj Waterfall?",
     "Indira Gandhi Park has a zoo, toy train, and musical fountain. Mirigikhoj Waterfall is suitable for trekking and nature photography."),
    (["religious_2", "nature_4"], "What distances from Rourkela are recorded for Vedvyas Temple and Pitamahal Dam?",
     "Vedvyas Temple is approximately 20 km from Rourkela. Pitamahal Dam is about 22 km from Rourkela."),
]

UNSUPPORTED = [
    "What is today's live queue time at Hanuman Vatika?",
    "What is the current weather forecast for Pitamahal Dam tomorrow?",
    "What is the latest 2026 crime rate at Koel Riverbank?",
    "How much is the current boating ticket at Mandira Dam?",
    "What is the phone number of Hotel Shubham?",
    "What time does Pizza Den close tonight?",
    "Which rooms are available at Hotel Radhika Regency right now?",
    "What is the live train fare from Delhi to Rourkela?",
    "What is the Wi-Fi password at Curry Pot?",
    "Which attractions in Delhi are described in this Rourkela knowledge base?",
]


def build():
    attractions = json.loads((ROOT / "rag.json").read_text(encoding="utf-8"))
    restaurants = json.loads((ROOT / "restaurants_rourkela.json").read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in attractions}
    cases = []

    def add(question, references, ids, category, records):
        cases.append({
            "id": f"rkl-{len(cases)+1:03d}", "question": question, "category": category,
            "answerable": bool(ids), "relevant_doc_ids": ids, "reference_answers": [references],
            "evidence": [{"doc_id": doc_id, "quote": json.dumps(record, ensure_ascii=False)}
                         for doc_id, record in zip(ids, records)],
        })

    for record_id, examples in ATTRACTIONS.items():
        for index, (question, reference) in enumerate(examples):
            add(question, reference, [f"rag.json:{record_id}"],
                "named_entity" if index < 3 else "paraphrase", [by_id[record_id]])
    for index in [0, 1, 2, 3, 4, 5, 7, 8, 10, 11]:
        row = restaurants[index]
        doc_id = f"restaurants_rourkela.json:{row['place_id']}"
        add(f"Where is {row['name']} located according to the dataset?",
            f"{row['name']} is at {row['address']}.", [doc_id], "named_entity", [row])
        add(f"What rating and review count are recorded for {row['name']}?",
            f"{row['name']} has a recorded rating of {row['rating']}/5 from {row['user_ratings_total']} reviews.",
            [doc_id], "named_entity", [row])
    for ids, question, reference in MULTI:
        add(question, reference, [f"rag.json:{record_id}" for record_id in ids],
            "multi_document", [by_id[record_id] for record_id in ids])
    for question in UNSUPPORTED:
        add(question, "INSUFFICIENT_CONTEXT", [], "unsupported", [])
    assert len(cases) == 100
    return {
        "schema_version": 1, "dataset_id": "rourkela-curated-rag", "dataset_version": "1.0.0",
        "origin": "curated",
        "description": "100 authored development cases, not users: 60 attraction questions, "
                       "20 restaurant questions, 10 multi-document questions, 10 unsupported questions. "
                       "References are derived from the bundled corpus, not externally fact-checked.",
        "corpus": [{"path": name, "sha256": json_sha256(ROOT / name)}
                   for name in ["rag.json", "restaurants_rourkela.json"]],
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/datasets/rourkela_v1.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Dataset already exists. Review labels and create a new version instead.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote 100 curated cases to {args.output.name}")
