import srsly
import typer
import warnings
from pathlib import Path

import spacy
from spacy.tokens import DocBin

def convert(lang: str, TRAIN_DATA, output_path: Path):
    nlp = spacy.blank(lang)
    db = DocBin()
    for text, annot in TRAIN_DATA:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)

TRAIN_DATA = [
              ('My favourite website is stackoverflow', {'entities': [(24,37,'WEBSITE')]}),
              ('My favourite website is github', {'entities': [(24,30,'WEBSITE')]}),
              ('My favourite website is instagram', {'entities': [(24,33,'WEBSITE')]}),
              ('My favourite website is liemcomputing', {'entities': [(24,37,'WEBSITE')]})]

convert("en" ,TRAIN_DATA, "./lol.spacy")