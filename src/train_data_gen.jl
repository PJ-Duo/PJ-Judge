module Spacy

using Conda
using PyCall

export spacy
const python3 = joinpath(Conda.python_dir(Conda.ROOTENV), "python3")
const spacy = pyimport("spacy")
const spacyTokens = pyimport("spacy.tokens")

function blank(model::String="en_core_web_md")
  spacy.blank(model)
end

function DocBin()
  return spacyTokens.DocBin()
end

end

nlp = Spacy.blank("en")
db = Spacy.DocBin()

training_data = [
  ("Tokyo Tower is 333m tall.", [(0, 11, "BUILDING")]),
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