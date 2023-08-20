### A Pluto.jl notebook ###
# v0.19.26

using Markdown
using InteractiveUtils

# ╔═╡ 13f35ab2-2f4f-4f93-95f4-f5043631da83
using DataFrames, CSV, LinearAlgebra, Crayons, BenchmarkTools, Random, Word2Vec, TextAnalysis, TextModels, StatsBase, Dates, JSON, BSON, Flux, EzXML, AbstractTrees, HTTP

# ╔═╡ b7198014-925c-4c55-b83a-7fe40898290a
# ╠═╡ skip_as_script = true
#=╠═╡
Threads.nthreads()
  ╠═╡ =#

# ╔═╡ df8ed368-6ddd-4d19-bb97-ed1077f55aef
@info "STARTUP => pkgs integrated. Initilizing..."

# ╔═╡ 83909f15-6987-496b-b614-9094cebd3a70
const STOPWORDS = Set(split(readlines(open("../data/stopwords/stopwords", "r"))[1]))

# ╔═╡ d8d1b080-6cb5-4fbf-a1b3-f5c59543eff9
const irregular_nouns = CSV.read("../data/grammer/irregular_nouns.csv", DataFrame)

# ╔═╡ 4b255ac9-5568-46f0-96b1-de1345e34334
function noun_to_singular(word::AbstractString)
    word = lowercase(word)
	
    if length(word) < 2
        return word
	elseif word in eachrow(irregular_nouns)
        return irregular_nouns[word]
	end
	
    if endswith(word, "s")
        if length(word) > 3
            # Case: thieves
            if endswith(word, "ves")
                if length(word[1:end-3]) > 2
                    return replace(word, "ves" => "f")
                else
                    return replace(word, "ves" => "fe")
				end
			end
            # Case: stories
            if endswith(word, "ies")
                return replace(word, "ies" => "y")
			end
            # Case: echoes
            if endswith(word, "es")
                if endswith(word, "ses") && word[end-3] in "aeiou"
        			return word[1:end-2]
			    end
                if endswith(word, "zzes")
                    return replace(word, "zzes" => "z")
				end
                return word[1:length(word)-2]
			end
            if endswith(word, "ys")
                return replace(word, "ys" => "y")
			end
            return word[1:end-1]
		end
    return word
	end
	return word
end

# ╔═╡ d8e9a7f7-8930-4181-b6e1-9947baf8e113
lemma_dict = Dict{String, Dict{String, String}}()

# ╔═╡ ac5b4ae2-7bff-405f-9f0c-24b444598d60
# filter the pos-grammer file & add it to a dict
open("../data/grammer/pos-grammer") do file
    for line in eachline(file)
        parts = split(line, "->")
        if length(parts) < 2
            continue
        end
        lemma = lowercase(strip(parts[1]))
        forms = split(parts[2], ",")
        for form in forms
            data = split(form, ">")
			for key in [strip(replace(data[1], "<" => ""))]
				word = lowercase(strip(data[2]))
            	lemma_dict[word] = get(lemma_dict, word, Dict())
            	lemma_dict[word][key] = lemma
    		end
        end
    end
end

# ╔═╡ cb9d1be7-1f36-4d26-a084-1f5c64354757
# achieve all tags from a file document and add into a dict
function nerify(file)
	dict = Dict{String, Vector{String}}()
    lines = readlines(file)
    n = length(lines)
    values = Vector{Vector{String}}(undef, n)
    keys = Vector{String}(undef, n)
    for i in 1:n
        line = split(lines[i])
        keys[i] = line[1]
        values[i] = line[2:end]
    end
    for i in 1:n
        dict[keys[i]] = [lowercase(ent) for ent in values[i]]
    end
	return dict
end

# ╔═╡ 9b511634-3d10-4f65-8c22-fdefd45b7116
function ner_tag(dict::Dict{String, Vector{String}}, search_term::String)
    for (tag, values) in dict
        if any(x -> search_term in split(x), values)
            return tag
        end
	end
end

# ╔═╡ 589917d6-aa38-4fda-96b1-4a0915895ffd
# checks if a word is a stopword => Returns a bool
function is_stopword(x)::Bool
    return x in STOPWORDS
end

# ╔═╡ 01321ada-fd71-4815-9ca8-bd0a5fbca3dc
# sentiment analysis pad sequences
function pad_sequences(l, maxlen=500)
    if length(l) <= maxlen
        res = zeros(maxlen - length(l))
        for ele in l
            push!(res, ele)
        end
        return res
    end
end

# ╔═╡ b41e9756-76fa-4402-9b3e-22b9ab183051
# sentiment analysis get embedding
function embedding(embedding_matrix, x)
    temp = embedding_matrix[:, Int64(x[1])+1]
    for i=2:length(x)
        temp = hcat(temp, embedding_matrix[:, Int64(x[i])+1])
    end
    return reshape(temp, reverse(size(temp)))
end

# ╔═╡ 9aae448f-d6a9-4779-af31-b27f0722b10d
function flatten(x)
    l = prod(size(x))
    x = permutedims(x, reverse(range(1, length=ndims(x))))
    return reshape(x, (l, 1))
end

# ╔═╡ baab8efa-e12d-403d-80ea-3b787d047456
@info "STARTUP => loading word vecs..."

# ╔═╡ dafe5b70-0192-401a-8cb7-cf9f47615b8a
# define & load pretrained word vectors
vec_model = wordvectors("../data/vecs/vec-pretrained")

# ╔═╡ 372dbfac-2aa8-4016-912c-5439e5969c08
# vectorizes a string of text
function vectorize(x)
	return mean([in_vocabulary(vec_model, word) ? get_vector(vec_model, word) : zeros(Float64, 100) for word in split(x)])
end

# ╔═╡ 2be833e6-23b7-411d-a069-8e9b4801e5f3
@info "STARTUP => loading vocabs..."

# ╔═╡ 903268a2-5dd9-43d2-824c-88cee7c0ce46
# define native vocab impl.
vocabn = Vocabulary(vocabulary(vec_model))

# ╔═╡ ee84f78e-c83f-4102-8cf7-64c2cd390ffc
# check if a word is out-of-vocab => Returns a bool
function is_oov(x)::Bool
	return !(x in keys(vocabn.vocab))
end

# ╔═╡ d6f94c81-37b6-400e-b9f3-78d6bb760f44
# rm out-of-vocab word(s) & chars
function rm_oov_punc(x)
    new_text = string()
    for (i, token) in enumerate(tokenize(x))
        if is_oov(token) || is_stopword(token)
            continue
        end
        new_text = string(new_text, token, " ")
    end
	return string(strip(new_text))
end

# ╔═╡ 172f1e31-5c64-4615-87fb-7c54bb1b35db
@info "STARTUP => loading pre-defind NER..."

# ╔═╡ 3b763a58-1c38-416b-9287-ab22b674472c
# define dict for named entitiy recognization
ner_model = nerify("../data/ner/ner")

# ╔═╡ 2a091f42-cc41-43f2-9055-c2b943dd3e79
@info "STARTUP => loading part-of-speech tagger..."

# ╔═╡ 780a5852-6c09-46f3-bc06-ff19c6caa112
# define part-of-speech tagger impl.
POStag_model = TextModels.PerceptronTagger(false)

# ╔═╡ 1875e69c-176f-4cf8-912a-8b1184f4f4d4
# actual lemmatize function
function lemmatize(sentence::AbstractString)
	tagged = TextModels.predict(POStag_model, sentence)
	final = string()
	for x in tagged
		if (x[2] == "NOUN")
			final = string(final, noun_to_singular(x[1]), " ")
			continue
		elseif x[1] in keys(lemma_dict) && x[2] in keys(lemma_dict[x[1]])
        	final = string(final, lemma_dict[x[1]][x[2]], " ")
			continue
		else
			final = string(final, x[1], " ")
			continue
		end
	end
	return strip(final)
end

# ╔═╡ c07999ef-09ad-49ee-b56d-c1712418eac1
# extracts nouns from a sentence using the POStagger
function nouns(sentence::AbstractString)
	final = Vector()
	for x in TextModels.predict(POStag_model, string(sentence))
		if (x[2] == "NOUN")
			push!(final, x[1])
		end
	end
	return final
end

# ╔═╡ 13ea869b-305d-4b0f-8e71-59dbf3dd7e9b
# train the POStag_model with a pretrained file
TextModels.fit!(POStag_model, [Tuple{String, String}[(string(words_and_tags[i]), string(words_and_tags[i+1])) for i=1:2:length(words_and_tags)-1] for words_and_tags in (split(sentence, " ") for sentence in readlines("../data/grammer/pos-pretrained"))])

# ╔═╡ 036ab902-ef1e-4725-ab4b-31eae9b88f08
@info "STARTUP => loading sentiment analysis..."

# ╔═╡ bf18bbac-a9f7-441a-acf8-a0843e000254
read_weights = BSON.load("../data/sentiment/model/sentiment-analysis-weights.bson")

# ╔═╡ e3507858-f88f-4b6d-9fe7-efefc2ad25b1
read_word_ids = JSON.parse(String(read(open("../data/sentiment/model/sentiment-analysis-word-to-id.json", "r"))))

# ╔═╡ bbdf222e-9e8b-41f3-9c8d-956db7554ff0
@info "STARTUP => Sentiment Analysis model trained with a $(length(read_word_ids)) word corpus!"

# ╔═╡ da5b7ebe-93c3-4d48-8e2b-cd74c8fa36be
function sensitize(x::AbstractString, out, unknowns=[])
	model = (x,) -> begin
    	a_1 = embedding(read_weights[:embedding_1]["embedding_1"]["embeddings:0"], x)
    	a_2 = flatten(a_1)
    	a_3 = Flux.Dense(read_weights[:dense_1]["dense_1"]["kernel:0"], read_weights[:dense_1]["dense_1"]["bias:0"], Flux.relu)(a_2)
    	a_4 = Flux.Dense(read_weights[:dense_2]["dense_2"]["kernel:0"], read_weights[:dense_2]["dense_2"]["bias:0"], Flux.sigmoid)(a_3)
    	return a_4
    end
    res = Array{Int, 1}()
    for ele in tokenize(x)
		if ele in keys(read_word_ids) && read_word_ids[ele] <= ( size(read_weights[:embedding_1]["embedding_1"]["embeddings:0"])[2] - 1)
            push!(res, read_word_ids[ele])
		else
			if !isempty(unknowns)
	    		for words in unknowns(ele)
					if words in keys(read_word_ids) && read_word_ids[words] <= size(read_weights[:embedding_1]["embedding_1"]["embeddings:0"])[2]
		    			push!(res, read_word_ids[words])
					end
			    end
			end
		end
    end
	threshold = 0.53
	if out == true
		threshold = 0.57
	end
	if model(pad_sequences(res))[1] >= threshold return "negative" else return "positive" end
end

# ╔═╡ 5f6b28f9-aba6-4f1d-a4f9-7fb29045c1d8
@info "STARTUP => loading the dataset..."

# ╔═╡ 8e7d3f84-6c3b-42a0-a05c-8f5926b557b2
dataset = dropmissing!(CSV.read("../data/dataset.csv", DataFrame))

# ╔═╡ 3fcab98d-9c9c-4981-94fd-ee1724e98e41
for i in size(dataset, 1):-1:1
    text = lowercase(dataset[i, 1])
    lemmatized = lemmatize(rm_oov_punc(text))
    if lemmatized == ""
        deleteat!(dataset, i)
    else
        dataset[i, 1] = lemmatized
    end
end

# ╔═╡ a64328d7-f0a8-40ec-b499-fcbc01db8d03
df_NERified_arr = [filter(!isnothing, [ner_tag(ner_model, ent) for ent in tokenize(row.query)]) for row in eachrow(dataset)]

# ╔═╡ 4c020d02-3e1d-4ddc-b37c-2da431c5ca19
df_NERified = dataset[setdiff(1:nrow(dataset), findall(x -> x == [], df_NERified_arr)), :]

# ╔═╡ 2d07cdec-bdce-4b3c-84fe-f5143d0de820
df_NERified_w_tags  = filter!(x -> !isempty(x), df_NERified_arr)

# ╔═╡ 9b543c72-1e9f-449b-8b45-2a51c4ae1a4c
print(Crayon(foreground = :green), Crayon(bold = true), "> ", Crayon(reset = true))

# ╔═╡ 2ad7bb7b-7152-4354-a5b1-ad003b71b2b1
startup = readline()

# ╔═╡ 4ba3d81d-365f-4670-af48-5c9a7ce1a7ae
function conclude_return(x)
	y = lemmatize(rm_oov_punc(lowercase(x)))

	if sensitize(y, false) == "negative"
		return "I don't think I'm allowed to answer that."
	end

	if isempty(strip(y))
		return "Sorry, I don't understand. Can you rephrase?"
	end

	# LAYER 1
	# ranking => (4/4) fastest and high accuracy
	# desc => filter by pre-defined NER tags
	ner_tags = filter(x -> !isnothing(x), [ner_tag(ner_model, ent) for ent in tokenize(y)])
	filtered_ds = df_NERified[setdiff(1:nrow(df_NERified), findall(x -> !issubset(x, ner_tags), df_NERified_w_tags)), :]

	# LAYER 2
	# ranking => (3/4) fast and accurate
	# note => this layer is dependent on layer 1's outcome
	# desc => filter by noun(s) within the input string (manipulating POStagger)
	# and then filter it further using countmaps
	input_nouns = nouns(y);
	filtered_ds = (nrow(filtered_ds) == 0) ? filter((row) -> issubset(input_nouns, split(row.query)), dataset) : filtered_ds
	y_countmap = collect(keys(countmap(tokenize(y))))
	filtered_ds = length(tokenize(y)) >= 2 ? filter(row -> all(contains(row.query, x) for x in y_countmap[1:2]), filtered_ds) : filter(row -> all(contains(row.query, x) for x in y_countmap), filtered_ds)

	if nrow(filtered_ds) == 0
		return "Sorry, I'm not trained enough to answer that."
	end
	
	#@info "Processing cosine similiarty..."
	sim_arr = Vector{Float64}()
	for row in eachrow(filtered_ds)
		x = row[1]
		try
			push!(sim_arr, dot(vectorize(x), vectorize(y)) / (norm(vectorize(x)) * norm(vectorize(y))))
		catch
			push!(sim_arr, 0.0)
		end
		
		if length(sim_arr) == nrow(filtered_ds)
			if maximum(sim_arr) >= 0.5
				output = filtered_ds[findfirst(x -> x == maximum(sim_arr), sim_arr), 2]
				#if sensitize(output, true) == "negative"
				#	return "I'm not quite sure how to answer that."
				#else
					return output
				#end
			else
				return "Sorry, I'm not sure if I have the right answer to that."
			end
		end
	end
end

# ╔═╡ bf60a9d5-4047-485e-9f91-a385703cd518
if lowercase(startup) == "chat"
	while true
		print(Crayon(foreground = :green), Crayon(bold = true), "query> ", Crayon(reset = true))
		chatInput = readline()

		if isempty(strip(chatInput))
			@info "=> Empty input string"
			continue
		end
	end
elseif lowercase(startup) == "exit"
	exit()
end

# ╔═╡ e46b816f-c8ec-429f-a5e9-832776dce6de
# ╠═╡ skip_as_script = true
#=╠═╡
conclude_return("founder of amazon")
  ╠═╡ =#

# ╔═╡ 0ff5fc4b-22c1-4cd4-aa66-2473cfe28be9
doc = parsehtml("""
<html>

<body>
    <p id="demo"></p>
    <a id="hello" href="https://google.com/">test</a>
    <script>
        x = 5;
        y = 10;
        z = x + y;
        window.alert(z);
    </script>
</body>
</html>""");

# ╔═╡ 37849ce6-45f3-45d3-b55e-b8cae3c1b19e
function checkJS(doc::EzXML.Document, username)
	scriptTag = findall("//script", doc)

	print(vectorize(nodecontent(doc.node)))
	
	if isempty(scriptTag) == false
		for tag in scriptTag
			if iselement(tag)
				return "hello"
			end
		end
	end
end

# ╔═╡ f02f4190-2c9d-4659-930e-06b7ef9b3e2a
checkJS(doc, "test")

# ╔═╡ 141c9ce7-602e-4e89-a60b-78c6a521d002
run(`echo test`)

# ╔═╡ 6a2be9d3-70f1-4ad8-8957-e7ba6285f263
WebSockets.open("ws://193.108.58.10:6969") do ws
	function getFinalMark(doc::EzXML.Document, username)
		totalMark = checkSkeleton(doc, username) + checkDuplicateIds(doc, username)
		WebSockets.send(ws, "{\"name\": \"" * username * "\", \"mark\": \"" * string(totalMark) * "/0.75\"}")
	end
	
	function checkSkeleton(doc::EzXML.Document, username)
		@info "Checking HTML Skeleton.."
		mark = 0.5
		tagMain = [if isempty(findall("//html", doc)) "html" else findall("//html", doc) end, if isempty(findall("//head", doc)) "head" else findall("//head", doc) end, if isempty(findall("//body", doc)) "body" else findall("//body", doc) end]
		for (i, subTagMain) in enumerate(tagMain)
			if subTagMain == "body" || subTagMain == "head" || subTagMain == "html"
				WebSockets.send(ws, "{\"name\": \"" * username * "\", \"message\": \"" * string(i) * ": Test Failed! ~ Missing " * subTagMain * " tag.\"" * "}")
				mark -= 0.25
				continue
			end
    		for tag in subTagMain
				if iselement(tag)
					WebSockets.send(ws, "{\"name\": \"" * username * "\", \"message\": \"" * string(i) * ": Test Passed.\"" * "}")
				end
			end
		end
		@info "Processed HTML Skeleton."
		return mark
	end

	function checkDuplicateIds(doc::EzXML.Document, username)
		@info "Checking for duplicate IDs.."
		mark = 0.25
		seen_ids = Set()
		for link in findall("//*[@id]/@id", doc)
			if link.content in seen_ids
				WebSockets.send(ws, "{\"name\": \"" * username * "\", \"message\": \"Duplicate element ID: " * link.content * "\"" * "}")
				mark -= 0.25
			else
				push!(seen_ids, link.content)
			end
		end
		@info "Processed checksum for duplicate IDs.."
		return mark
	end
	
    for msg in ws
		getFinalMark(parsehtml(JSON.parse(msg)["data"]), JSON.parse(msg)["name"])
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractTrees = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Crayons = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
EzXML = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
TextAnalysis = "a2db99b7-8b79-58f8-94bf-bbc811eef33d"
TextModels = "77b9cbda-2a23-51df-82a3-24144d1cd378"
Word2Vec = "c64b6f0f-98cd-51d1-af78-58ae84944834"

[compat]
AbstractTrees = "~0.3.4"
BSON = "~0.3.7"
BenchmarkTools = "~1.3.2"
CSV = "~0.10.11"
Crayons = "~4.1.1"
DataFrames = "~1.3.6"
EzXML = "~1.1.0"
Flux = "~0.12.8"
HTTP = "~1.9.6"
JSON = "~0.21.4"
StatsBase = "~0.33.21"
TextAnalysis = "~0.7.4"
TextModels = "~0.1.1"
Word2Vec = "~0.5.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "3ef0b23eae8c7f0a9df51281c3dc4a14504dd681"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "76289dc51920fdc6e0013c872ba9551d54961c24"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.2"

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

    [deps.Adapt.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "1ee88c4c76caa995a885dc2f22a5d548dfbbc0ba"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.BSON]]
git-tree-sha1 = "2208958832d6e1b59e49f53697483a84ca8d664e"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "d9a9701b899b30332bbcb3e1679c41cce81fb0e8"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.2"

[[deps.BitFlags]]
git-tree-sha1 = "43b1a4a8f797c1cddadf60499a8a077d4af2cd2d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.7"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "44dbf560808d49041989b8a96cae4cffbeb7966a"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.11"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "6717cb9a3425ebb7b31ca4f832823615d175f64a"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.13.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "61549d9b52c88df34d21bd306dba1d43bb039c87"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.51.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "6c0100a8cf4ed66f66e2039af7cde3357814bad2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.46.2"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "96d823b94ba8d187a6d8f0826e731195a74b90e9"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.2.0"

[[deps.CorpusLoaders]]
deps = ["CSV", "DataDeps", "Glob", "InternedStrings", "LightXML", "MultiResolutionIterators", "StringEncodings", "WordTokenizers"]
git-tree-sha1 = "01a12a78eca5da25b95a661716f4416d4264bced"
uuid = "214a0ac2-f95b-54f7-a80b-442ed9c2c9e8"
version = "0.3.5"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "bc0a264d3e7b3eeb0b6fc9f6481f970697f29805"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.10"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "0fa3b52a04a4e210aeb1626def9c90df3ae65268"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.1.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "e17cc4dc2d0b0b568e80d937de8ed8341822de67"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.2.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["AbstractTrees", "Adapt", "ArrayInterface", "CUDA", "CodecZlib", "Colors", "DelimitedFiles", "Functors", "Juno", "LinearAlgebra", "MacroTools", "NNlib", "NNlibCUDA", "Pkg", "Printf", "Random", "Reexport", "SHA", "SparseArrays", "Statistics", "StatsBase", "Test", "ZipFile", "Zygote"]
git-tree-sha1 = "e8b37bb43c01eed0418821d1f9d20eca5ba6ab21"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.12.8"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

    [deps.ForwardDiff.weakdeps]
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Functors]]
git-tree-sha1 = "223fffa49ca0ff9ce4f875be001ffe173b2b7de4"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.8"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "a3351bc577a6b49297248aadc23a4add1097c2ac"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.7.1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "19d693666a304e8c371798f4900f7435558c7cde"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.17.3"

[[deps.Glob]]
git-tree-sha1 = "97285bbd5230dd766e9ef6749b80fc617126d496"
uuid = "c27321d9-0574-5035-807b-f59d2c89b15c"
version = "1.3.1"

[[deps.HTML_Entities]]
deps = ["StrTables"]
git-tree-sha1 = "c4144ed3bc5f67f595622ad03c0e39fa6c70ccc7"
uuid = "7693890a-d069-55fe-a829-b4a6d304f0ee"
version = "1.0.1"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e77dbf117412d4f164a464d610ee6050cc75272"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.9.6"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "4ced6667f9974fc5c5943fa5e2ef1ca43ea9e450"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.8.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "f044a2796a9e18e0531b9b3072b0019a61f264bc"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.17.1"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "070e4b5b65827f82c16ae0916376cb47377aa1b5"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.18+0"

[[deps.Languages]]
deps = ["InteractiveUtils", "JSON"]
git-tree-sha1 = "b1a564061268ccc3f3397ac0982983a657d4dcb8"
uuid = "8ef0a80b-9436-5d2c-a485-80b904378c43"
version = "0.4.3"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c7cb1f5d892775ba13767a87c7ada0b980ea0a71"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+2"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "e129d9391168c677cd4800f5c0abb1ed8cb3794f"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "c3ce8e7420b3a6e071e0fe4745f5d4300e37b13f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.24"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "cedb76b37bc5a6c702ade66be44f831fa23c681e"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultiResolutionIterators]]
deps = ["IterTools", "Random", "Test"]
git-tree-sha1 = "27fa99913e031afaf06ea8a6d4362fd8c94bb9fb"
uuid = "396aa475-d5af-5b65-8c11-5c82e21b2380"
version = "0.5.0"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "3a8dfd0cfb5bb3b82d09949e14423409b9334acb"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.34"

[[deps.NNlibCUDA]]
deps = ["CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "a2dc748c9f6615197b6b97c10bcce829830574c9"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.1.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cae3153c7f6cf3f069a853883fd1919a6e5bab5b"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.9+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5a6ab2f64388fd1175effdf73fe5933ef1e0bac0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.7.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "9673d39decc5feece56ef3940e5dafba15ba0f81"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "7eb1686b4f04b82f96ed7a4ea5890a4f0c7a09f1"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "04bdff0b09c65ff3e06a05e3eb7b120223da3d39"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Snowball]]
deps = ["Languages", "Snowball_jll", "WordTokenizers"]
git-tree-sha1 = "d38c1ff8a2fca7b1c65a51457dabebef28052399"
uuid = "fb8f903a-0164-4e73-9ffe-431110250c3b"
version = "0.1.0"

[[deps.Snowball_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ff3a185a583dca7265cbfcaae1da16aa3b6a962"
uuid = "88f46535-a3c0-54f4-998e-4320a1339f51"
version = "2.2.0+0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "c60ec5c62180f27efea3ba2908480f8055e17cee"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "7f5a513baec6f122401abfc8e9c074fdac54f6c1"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.1"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "45a7769a04a3cf80da1c1c7c60caf932e6f4c9f7"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.6.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StrTables]]
deps = ["Dates"]
git-tree-sha1 = "5998faae8c6308acc25c25896562a1e66a3bb038"
uuid = "9700d1a9-a7c8-5760-9816-a99fda30bb8f"
version = "1.0.1"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "33c0da881af3248dafefb939a21694b97cfece76"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.6"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TextAnalysis]]
deps = ["DataStructures", "DelimitedFiles", "JSON", "Languages", "LinearAlgebra", "Printf", "ProgressMeter", "Random", "Serialization", "Snowball", "SparseArrays", "Statistics", "StatsBase", "Tables", "WordTokenizers"]
git-tree-sha1 = "e60c42c140e130873e18ef1201d22401b54062cd"
uuid = "a2db99b7-8b79-58f8-94bf-bbc811eef33d"
version = "0.7.4"

[[deps.TextModels]]
deps = ["BSON", "CUDA", "CorpusLoaders", "DataDeps", "DataStructures", "DelimitedFiles", "Flux", "JSON", "Languages", "NNlib", "Pkg", "Random", "StatsBase", "Test", "TextAnalysis", "WordTokenizers", "Zygote"]
git-tree-sha1 = "13afea4360a5094899ed9e690f11e18ea72273bc"
uuid = "77b9cbda-2a23-51df-82a3-24144d1cd378"
version = "0.1.1"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Word2Vec]]
deps = ["LinearAlgebra", "Statistics", "Word2Vec_jll"]
git-tree-sha1 = "a4e76aeaaf2bda1556864b610051960cea642958"
uuid = "c64b6f0f-98cd-51d1-af78-58ae84944834"
version = "0.5.3"

[[deps.Word2Vec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "264768df753f8328295d7b7cff55edc52f180284"
uuid = "9fbe4022-c126-5389-b4b2-756cc9f654d0"
version = "0.1.0+0"

[[deps.WordTokenizers]]
deps = ["DataDeps", "HTML_Entities", "StrTables", "Unicode"]
git-tree-sha1 = "01dd4068c638da2431269f49a5964bf42ff6c9d2"
uuid = "796a5d58-b03d-544a-977e-18100b691f6e"
version = "0.5.6"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "93c41695bc1c08c46c5899f4fe06d6ead504bb73"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.10.3+0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "3593e69e469d2111389a9bd06bac1f3d730ac6de"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.4"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "5be3ddb88fc992a7d8ea96c3f10a49a7e98ebc7b"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.62"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═13f35ab2-2f4f-4f93-95f4-f5043631da83
# ╠═b7198014-925c-4c55-b83a-7fe40898290a
# ╠═df8ed368-6ddd-4d19-bb97-ed1077f55aef
# ╠═83909f15-6987-496b-b614-9094cebd3a70
# ╠═d8d1b080-6cb5-4fbf-a1b3-f5c59543eff9
# ╠═4b255ac9-5568-46f0-96b1-de1345e34334
# ╠═d8e9a7f7-8930-4181-b6e1-9947baf8e113
# ╠═ac5b4ae2-7bff-405f-9f0c-24b444598d60
# ╠═1875e69c-176f-4cf8-912a-8b1184f4f4d4
# ╠═c07999ef-09ad-49ee-b56d-c1712418eac1
# ╠═cb9d1be7-1f36-4d26-a084-1f5c64354757
# ╠═9b511634-3d10-4f65-8c22-fdefd45b7116
# ╠═ee84f78e-c83f-4102-8cf7-64c2cd390ffc
# ╠═589917d6-aa38-4fda-96b1-4a0915895ffd
# ╠═d6f94c81-37b6-400e-b9f3-78d6bb760f44
# ╠═01321ada-fd71-4815-9ca8-bd0a5fbca3dc
# ╠═b41e9756-76fa-4402-9b3e-22b9ab183051
# ╠═9aae448f-d6a9-4779-af31-b27f0722b10d
# ╠═372dbfac-2aa8-4016-912c-5439e5969c08
# ╠═baab8efa-e12d-403d-80ea-3b787d047456
# ╠═dafe5b70-0192-401a-8cb7-cf9f47615b8a
# ╠═2be833e6-23b7-411d-a069-8e9b4801e5f3
# ╠═903268a2-5dd9-43d2-824c-88cee7c0ce46
# ╠═172f1e31-5c64-4615-87fb-7c54bb1b35db
# ╠═3b763a58-1c38-416b-9287-ab22b674472c
# ╠═2a091f42-cc41-43f2-9055-c2b943dd3e79
# ╠═780a5852-6c09-46f3-bc06-ff19c6caa112
# ╠═13ea869b-305d-4b0f-8e71-59dbf3dd7e9b
# ╠═036ab902-ef1e-4725-ab4b-31eae9b88f08
# ╠═bbdf222e-9e8b-41f3-9c8d-956db7554ff0
# ╠═bf18bbac-a9f7-441a-acf8-a0843e000254
# ╠═e3507858-f88f-4b6d-9fe7-efefc2ad25b1
# ╠═da5b7ebe-93c3-4d48-8e2b-cd74c8fa36be
# ╠═5f6b28f9-aba6-4f1d-a4f9-7fb29045c1d8
# ╠═8e7d3f84-6c3b-42a0-a05c-8f5926b557b2
# ╠═3fcab98d-9c9c-4981-94fd-ee1724e98e41
# ╠═a64328d7-f0a8-40ec-b499-fcbc01db8d03
# ╠═4c020d02-3e1d-4ddc-b37c-2da431c5ca19
# ╠═2d07cdec-bdce-4b3c-84fe-f5143d0de820
# ╠═9b543c72-1e9f-449b-8b45-2a51c4ae1a4c
# ╠═2ad7bb7b-7152-4354-a5b1-ad003b71b2b1
# ╠═4ba3d81d-365f-4670-af48-5c9a7ce1a7ae
# ╠═bf60a9d5-4047-485e-9f91-a385703cd518
# ╠═e46b816f-c8ec-429f-a5e9-832776dce6de
# ╠═0ff5fc4b-22c1-4cd4-aa66-2473cfe28be9
# ╠═37849ce6-45f3-45d3-b55e-b8cae3c1b19e
# ╠═f02f4190-2c9d-4659-930e-06b7ef9b3e2a
# ╠═141c9ce7-602e-4e89-a60b-78c6a521d002
# ╠═6a2be9d3-70f1-4ad8-8957-e7ba6285f263
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
