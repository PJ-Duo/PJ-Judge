import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
TRAIN_DATA = [('what is the price of polo?', {'entities': [(21, 25, 'PrdName')]}), 
              ('what is the price of ball?', {'entities': [(21, 25, 'PrdName')]}), 
              ('what is the price of jegging?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of t-shirt?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of jeans?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of bat?', {'entities': [(21, 24, 'PrdName')]}), 
              ('what is the price of shirt?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of bag?', {'entities': [(21, 24, 'PrdName')]}), 
              ('what is the price of cup?', {'entities': [(21, 24, 'PrdName')]}), 
              ('what is the price of jug?', {'entities': [(21, 24, 'PrdName')]}), 
              ('what is the price of plate?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of glass?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of moniter?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of desktop?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of bottle?', {'entities': [(21, 27, 'PrdName')]}), 
              ('what is the price of mouse?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of keyboad?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of chair?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of table?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of watch?', {'entities': [(21, 26, 'PrdName')]}),
              ('My favourite website is youtube', {'entities': [(23,30,'WEBSITE')]})
              ('My favourite website is stackoverflow', {'entities': [(23,36,'WEBSITE')]})
              ('My favourite website is github', {'entities': [(23,29,'WEBSITE')]})
              ('My favourite website is instagram', {'entities': [(23,31,'WEBSITE')]})
              ('My favourite website is liemcomputing', {'entities': [(23,36,'WEBSITE')]})]
# the DocBin will store the example documents
db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./train.spacy")