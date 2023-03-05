### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 13f35ab2-2f4f-4f93-95f4-f5043631da83
using DataFrames, CSV, LinearAlgebra, JSON, 
Crayons

# ╔═╡ 0941c3fc-bac8-11ed-11d5-6318de0d8aec
module Spacy

using Conda
using PyCall

export spacy
const python3 = joinpath(Conda.python_dir(Conda.ROOTENV), "python3")
const spacy = pyimport("spacy")
const spacyMatcher = pyimport("spacy.matcher")
const spacyUtil = pyimport("spacy.util")

function load(model::String="en_core_web_sm")
    try
        spacy.load(model)
    catch ex
        if isa(ex, PyError) && getfield(ex, :exception) == PyObject(pyimport("ImportError"))
            run(`$python3 -m spacy download $model`)
            spacy.load(model)
        else
            rethrow()
        end
    end
end

# Define language
module language
using PyCall
const spacyLang = pyimport("spacy.language")
function from_config(x)
    spacyLang.Language.from_config(x)
end
function component(x, y)
    spacyLang.Language.component(x, func=y)
end
end

# Define training
module training
using PyCall
const spacyLang = pyimport("spacy.language")
function from_config(x)
    spacyLang.Language.from_config(x)
end
function component(x, y)
    spacyLang.Language.component(x, func=y)
end
end

function matcher(nlp)
    spacyMatcher.matcher(nlp)
end

function matcher_add(x, y, z)
    if isnothing(z)
        spacyMatcher.matcher.add(x, y, on_match=z)
    else
        spacyMatcher.matcher.add(x, y)
    end
end

function matcher_remove(x)
    spacyMatcher.matcher.remove(x)
end

function matcher_get(x)
    return spacyMatcher.matcher.get(x)
end
end


# ╔═╡ 76e1dfb9-b953-422c-a06a-9d87027f9c3b
dataset = CSV.read("../data/dataset.csv", DataFrame)

# ╔═╡ e55d111b-333f-4715-89eb-5336f06c21e7
nlp = Spacy.load("en_core_web_md")

# ╔═╡ a81bbd73-5dbf-4711-bd5a-bcb716560cfd
patterns = [
	Dict("label" => "COMPUTER", "pattern" => [Dict("lower" => "notebook")]),
	Dict("label" => "CURRENCY", "pattern" => [Dict("lower" => "francs")]),
	Dict("label" => "PART", "pattern" => [Dict("lower" => "disk")]),
	Dict("label" => "PROGLANG", "pattern" => [
		Dict("lower" => "java"), 
		Dict("lower" => "python"), 
		Dict("lower" => "c++")]
	)
]

# ╔═╡ e43bf442-39dc-44fc-b631-85402db7ddec
ruler = nlp.add_pipe("entity_ruler")

# ╔═╡ 67e05f07-2829-4d17-bf04-ae5ca3d70c62
ruler.add_patterns(patterns)

# ╔═╡ 6bfedfbb-399c-40f6-a4e1-852ffb6f38f6
# ╠═╡ skip_as_script = true
#=╠═╡
lol = nlp(lowercase("""Hello I would like to order a notebook with 16GB and 256 GB disk, I would like to spend less than 1000 dollars, what would be the options. I also use Java and C++ and python as programming languages. Thanks a lot
Porya"""))
  ╠═╡ =#

# ╔═╡ 99866c59-b08e-4892-9765-dc187dc209d1
# ╠═╡ skip_as_script = true
#=╠═╡
for entity in lol.ents      
     println("$(entity.text) ($(entity.label_))")
end
  ╠═╡ =#

# ╔═╡ 9b543c72-1e9f-449b-8b45-2a51c4ae1a4c
print(Crayon(foreground = :green), Crayon(bold = true), "> ", Crayon(reset = true))

# ╔═╡ 2ad7bb7b-7152-4354-a5b1-ad003b71b2b1
startup = readline()

# ╔═╡ bc0847ef-dd14-4283-93b7-d226950a2798
arr = []

# ╔═╡ bf60a9d5-4047-485e-9f91-a385703cd518
if lowercase(startup) == "chat"
	while true
		print(Crayon(foreground = :green), Crayon(bold = true), "query> ", Crayon(reset = true))
		chatInput = readline()

		if lowercase(chatInput) == "exit"
			break
		end
		
		for (i, row) in enumerate(eachrow( dataset ))
			x = nlp(lowercase(row[1]))
			y = nlp(lowercase(chatInput))

			for token in y
				if token.is_oov 
					y = nlp(replace(y.text, token.text => ""))
				end
			end
	
			x_vector = x.vector
			y_vector = y.vector
	
			push!(arr, dot(x_vector, y_vector) / (norm(x_vector) * norm(y_vector)))

			if nrow(dataset) == i
				if maximum(arr) >= 0.6
					println(Crayon(foreground = :red), Crayon(bold = true), "return> ", Crayon(reset = true), dataset[findfirst(x -> x == maximum(arr), arr), 2])
					println(arr)
					empty!(arr)
					break;
				else
					println(Crayon(foreground = :red), Crayon(bold = true), "return> ", Crayon(reset = true), "Sorry, I'm not trained enough to answer that question.")
					empty!(arr)
					break;
				end
			end
		end
	end
elseif lowercase(startup) == "exit"
	exit()
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
Conda = "8f4d0f93-b110-5947-807f-2305c1781a2d"
Crayons = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"

[compat]
CSV = "~0.10.9"
Conda = "~1.8.0"
Crayons = "~4.1.1"
DataFrames = "~1.5.0"
JSON = "~0.21.3"
PyCall = "~1.95.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "5786f4fed66c09a3f8f76da1c45ca6a841241f53"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "SnoopPrecompile", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "c700cce799b51c9045473de751e9319bdd1c6e94"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.9"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "e32a90da027ca45d84678b826fffd3110bb3fc90"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.8.0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

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

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "96f6db03ab535bdb901300f88335257b0018689d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "62f417f6ad727987c755549e9cd88c46578da562"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.95.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"
"""

# ╔═╡ Cell order:
# ╠═13f35ab2-2f4f-4f93-95f4-f5043631da83
# ╠═0941c3fc-bac8-11ed-11d5-6318de0d8aec
# ╠═76e1dfb9-b953-422c-a06a-9d87027f9c3b
# ╠═e55d111b-333f-4715-89eb-5336f06c21e7
# ╠═a81bbd73-5dbf-4711-bd5a-bcb716560cfd
# ╠═e43bf442-39dc-44fc-b631-85402db7ddec
# ╠═67e05f07-2829-4d17-bf04-ae5ca3d70c62
# ╠═6bfedfbb-399c-40f6-a4e1-852ffb6f38f6
# ╠═99866c59-b08e-4892-9765-dc187dc209d1
# ╠═9b543c72-1e9f-449b-8b45-2a51c4ae1a4c
# ╠═2ad7bb7b-7152-4354-a5b1-ad003b71b2b1
# ╠═bc0847ef-dd14-4283-93b7-d226950a2798
# ╠═bf60a9d5-4047-485e-9f91-a385703cd518
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002