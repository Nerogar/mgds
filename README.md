# MGDS

MGDS is a custom dataset implementation for Pytorch that is built around the idea of a node based graph.

Each node in the graph can take inputs from previous nodes, do various calculations, and return outputs to nodes further
down the chain. This modular approach enables highly complex data processing pipelines only by plugging together
pre-built modules.

This project is currently developed for the use in [OneTrainer](https://github.com/Nerogar/OneTrainer), but it can be
used in all kinds of different applications. 