## Data(s) Used by PJ_Judge

For copyright, privacy concerns and the enormous size, we've decided to avoid disclosing corpus, datasets, and pretrained vecs. 

Although, we provide an open-source corpus constructor out of dump of a wiki:
- [wiki-corpus](https://github.com/PJ-Duo/wiki-corpus) `=>` Creates a wikipedia text corpus using a wiki dump file

Ensure to place the final corpus inside of the `cprs` folder under the name `corpus`.

Furthermore, the `dataset.csv` file contains questions with answers to them. The larger it is, the more knowledge PJ_Judge will have. You just need to ensure it is in a `csv` format and has both query and ans columns field. You can use services such as discord (their reply feature), twitter or etc to scrap tons of data. We've created a scrapper for discord, alongisde a data-cleaning tool in Java:
- [discord-scraper](https://github.com/PJ-Duo/discord-scraper) `=>` Scraps a discord channel with the data being in "message and reply to the message" format
- [JSON-to-CSV-Data-Cleaner](https://github.com/PJ-Duo/JSON-to-CSV-Data-Cleaner) `=>` Converts JSON data to CSV format while filtering and cleaning the data

Lastly, ensure you have word vectors. For getting word vectors you need to have a large corpus with alot of words (like the wiki method mentioned at the beginning), and then use [Word2Vec.jl](https://github.com/JuliaText/Word2Vec.jl) to convert the large corpus into vectors. Remember, the more words converted to vectors, the more accurate when it comes to processing. After you're done with converting, place the file inside the `vecs` folder under the name `vec-pretrained`.


Please note that the files inside of the `vecs` and `crps` are just placeholders and are required to be replaced.
