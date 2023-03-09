module Spacy

using Conda
using PyCall

export spacy
const python3 = joinpath(Conda.python_dir(Conda.ROOTENV), "python3")
const spacy = pyimport("spacy")
const spacyTokens = pyimport("spacy.tokens")

function blank(model::String="en_core_web_sm")
  spacy.blank(model)
end

function DocBin()
  return spacyTokens.DocBin()
end

end

nlp = Spacy.blank("en")
db = Spacy.DocBin()

training_data = [
  ("My favourite website is Stack Overflow", [(24,38,"WEBSITE")]),
  ("My favourite website is github", [(24,30,"WEBSITE")]),
  ("My favourite website is instagram", [(24,33,"WEBSITE")]),
  ("My favourite website is liemcomputing", [(24,37,"WEBSITE")]),
]


for (text, annotations) in training_data
  doc = nlp(text)
  ents = []
  for (start, last, label) in annotations
    span = doc.char_span(start, last, label=label)
    push!(ents, span)
  end
  doc.ents = ents
  db.add(doc)
end

db.to_disk("./train.spacy")