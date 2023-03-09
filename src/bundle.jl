using PackageCompiler

pluto = open("pluto.jl", "r")
output = open("_temp.jl", "w")

for (i, line) in enumerate(eachline(pluto))
    if i == 1
        write(output, "module PJ_Judge")
    end
    if !startswith(line, "#") && !startswith(line, "using Markdown") && !startswith(line, "using InteractiveUtils")
        if startswith(line, "PLUTO_PROJECT_TOML_CONTENTS")
            write(output, "end")
            break
        else 
            write(output, line * "\n")
        end
    end
end

close(pluto)
close(output)

create_app("..", "build", force=true)

rm("_temp.jl")