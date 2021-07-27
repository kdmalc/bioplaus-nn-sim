# hh-comp
## Research Title: 
## "Dynamical Patterns of Neural Models in Hodgkin-Huxley Form"
Embedding Hodgkin-Huxley model understanding into computational frameworks for Brown 2021

Notebook Level Descriptions <br/>
0X - Base NBs for demonstrations/learning for various packages used <br/>
1X - Foundational simulations for single entities (not yet networks) <br/>
2X - Intro to networks and procedural generation of such networks <br/>
3X - Results/Figures: Single and 10 NN cases, parameter/coupling variations <br/>

## Abstract:
The famous Hodgkin-Huxley (HH) system of differential equations (ODEs) describes electrochemical dynamics of the squid giant axon underlying action potential initiation and propagation. Since their introduction in the 1950’s, many generalizations of the ODEs have been used to model a diverse range of neural systems. To facilitate investigation of the essential dynamics of such “HH-type” models, this project seeks to develop a software tool (in python) that procedurally generates networks of HH-type blocks, with configurable parameters and network architecture. Here the focus is not on biological plausibility per se, but the structure of dynamical types imposed by the defining structure of HH-type ODEs. We seek to classify the qualitative behavior of simulated network behaviors, to guide dimensionality reduction algorithms and develop efficient control strategies. An ongoing focus of this project is to determine a way to consistently measure such complexity so that these groupings are informative for inference of unobserved network structures. In future work, we will use this platform to computationally define control schemes based on probe stimuli of the network.
