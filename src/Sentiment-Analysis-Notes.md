Dev-notes (WIP)

Deps:

CorpusLoaders

Then run the following in Julia terminal:

using CorpusLoaders
dataset_test_pos = load(IMDB("test_pos"))
dataset_train_neg = load(IMDB("train_neg"))
dataset_test_neg = load(IMDB("test_neg"))

using Base.Iterators
docs = collect(take(dataset_train_pos, 2) #This will take awhile...