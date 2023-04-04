import Pkg; Pkg.add("Pluto")
import Pluto; Pluto.run(threads=trunc(Int, length(Sys.cpu_info())/2))