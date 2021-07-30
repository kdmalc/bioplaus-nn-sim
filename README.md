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

## Notes to Self to Update / Reminders From Refactoring
1. Note that the current axes (not in NB 32 and beyond) of the gating variables is "Voltage" or "Membrane Poential", and, as it ranges from 0 to 1, should really be "Open State" or more generally, "Openess", as the gating variables just indicate how open the protein-gated ion channel is.
2. The sigmoid approximations do not hold when applying current for long time intervals, more spikes appear in the approximation than in the "true" case.
3. The conductivity matrices may or may not still be broken: jury is still out.  They return non-neural activity, but this isn't necessarily incorrect.
4. The numerical solver, odeint, appears to not solve at every single time point.  I've come to this conclusion based on the fact that the applied current does not always show up in all runs.  Furthermore, making it print the applied current at all times, there are times when the current should be 0.1 but it never appears.  This should be double-checked.  I was able to get around this by specificying continuous current, for say 20 ms instead of 2 ms, which ensured that the desired time interval recieved the desired current.
