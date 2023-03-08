# Import Discord.jl.
using Discord
using DotEnv
DotEnv.config()
println(ENV["token"])
# Create a client.


c = Client("MTA4MzA3OTk3MjI5OTc0NzM0OQ.GrSuxg.F7uNCw5aA9h3yxjh4Pcjc68KfEX1YKYMKlr9QI"; presence=(game=(name="with Discord.jl", type=AT_GAME),))

# Create a handler for the MessageCreate event.
function handler(c::Client, e::MessageCreate)
    # Display the message contents.
    println("Received message: $(e.message.content)")
    # Add a reaction to the message.
    create(c, Reaction, e.message, '👍')
end

# Add the handler.
add_handler!(c, MessageCreate, handler)
# Log in to the Discord gateway.
open(c)
# Wait for the client to disconnect.
wait(c)