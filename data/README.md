## Data(s) used by PJ_Judge

For copyright, privacy concerns and the enormous size, we've decided to avoid disclosing corpus, datasets, and pretrained vecs. 

Although, we provide an open-source corpus constructor out of dump of a wiki:
- [wiki-corpus](https://github.com/PJ-Duo/wiki-corpus) `=>` Creates a wikipedia text corpus using a wiki dump file

Ensure to place the final corpus inside of the `cprs` folder under the name `corpus`.

Furthermore, the `dataset.csv` file contains questions with answers to them. The larger it is, the more knowledge PJ_Judge will have. You just need to ensure it is in a `csv` format and has both query and ans columns field. You can use services such as discord (their reply feature), twitter or etc to scrap tons of data. We've created a scrapper for discord, alongisde a data-cleaning tool in Java:
- [discord-scraper](https://github.com/PJ-Duo/discord-scraper) `=>` Scraps a discord channel with the data being in "message and reply to the message" format
- [JSON-to-CSV-Data-Cleaner](https://github.com/PJ-Duo/JSON-to-CSV-Data-Cleaner) `=>` Converts JSON data to CSV format while filtering and cleaning the data

Lastly, ensure you have word vectors. For getting word vectors you need to have a large corpus with alot of words (like the wiki method mentioned at the beginning), and then using [Word2Vec.jl](https://github.com/JuliaText/Word2Vec.jl) to convert the large corpus into vectors. Remember, the more words converted to vectors, the more accurate when it comes to processing, and the more words included in the AI's vocab. After you're done with converting, place the file inside the `vecs` folder under the name `vec-pretrained`.


Please note that the files inside of `vecs` and `crps` are just placeholders and are meant to be replaced. Also, the `crps` folder doesn't get used inside of the SourceCode directly, it is used for training externally only, and for organization purposes.


## Data(s) provided with PJ_Judge

The `grammer` and `ner` directories are both provided by PJ_Judge. 

The grammer folder contains three directories: `pos-grammer`, `pos-pretrained`, and `irregular_nouns.csv`. `pos-grammer` being a corpus of Part-of-Speech (POS) grammers and is a list of lemmas and variations extracted from the British National Corpus (BNC). `pos-pretrained`is a pretrained file that includes sentences with their POS tag assigned, and is used to train the POS-tagger. `irregular_nouns.csv` is a list of irregular nouns and their singular form used for handling the case of plural nouns.

The `ner` folder contains a `ner` file and the file is used for named entity recognition, which helps majorly when it comes to decision making of PJ_Judge. Pay close attention to the format, but to sum it up, the first word in each line represents the tag name, and the rest following with a space are the entities of the tag. 

