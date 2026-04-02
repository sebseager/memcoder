## Ideas for what to turn into lora


1. Bug fix trajectories
    Have some representation of the model trying and succeeding to solve various 
    challenges. This teaches it how to format tool calls correctly, patterns that 
    tend to work, etc. You have the regular model (or maybe a stronger teacher model) 
    generate trajectories of tool calls. Then put those into the hypernetwork and 
    do weight updates based on that. That way the model "learns" what trajectories 
    or patterns tend to work better.
    - Negative "award" for stuff that didn't work

2. Static chunks (easy)
    - Take with some granularity chunks from the repo, toss them into HN, get LoRAs
    - Store a mapping between code areas and their associated LoRA

3. Dynamic chunks (slightly harder)
    - Have agent explore chunk
    - Agent then writes a design document on the chunk, how it works, its conventions, and how it relates to other chunks
    - This design doc then gets turned into a lora

4. 
    
## TODO

1. What goes in the input to the LORA?
2. How is the output managed and accessed?