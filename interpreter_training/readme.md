# How to train

We have unruly prompt lengths (300k tokens in initial testing). Had to use deepspeed ulysses for multi-gpu sequence parallelization, and that wasn't enough so we ended up using tiling too ([arctic training](https://www.snowflake.com/en/engineering-blog/arctic-long-sequence-training-multi-million-token-ai/) - [github guide](https://github.com/snowflakedb/ArcticTraining/tree/main/projects/sequence-parallelism)).

After a lot of fiddling I got it working on an 8xA100 instance (with plenty of room, only about 25% gpu utilization). The `setup.sh` script has all the steps I went through to get it working. 

Note: I'm not using flash attention right now, just sdpa. I want to get flash attention working later but had a bunch of environment issues that were upsetting me.


If anyone is reading this these are just prelim notes for myself, I'll do a better job documenting stuff once I get deeping into the project. 

ssh -i lambdalabs.pem ubuntu@170.9.244.20
