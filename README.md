# hh-comp
## Research Title: 
## "Dynamical Patterns of Neural Models in Hodgkin-Huxley Form"
Embedding Hodgkin-Huxley model understanding into computational frameworks for Brown 2021

Notebook Level Descriptions
0X - Base NBs for demonstrations/learning for various packages used \n
1X - Foundational simulations for single entities (not yet networks) \n
2X - Intro to networks and procedural generation of such networks \n

## 60 Second Background:
This project is focused on data dimensionality reduction for neuroscience.  Overcoming the curse of dimensionality and removing irrelevant features from data has become increasingly necessary in neuroscience, in order to lower computational costs, avoid overfitting, and isolate relevant features. 

In higher order spaces, datapoints may appear to be equidistant, making it difficult to identify relationships, especially for machine learning models which typically use distance in space as a proxy for similarity (i.e. knn).

Hodgkin-Huxley-like differential equations model the current flows through neurons, and these equations form a dynamical system that can be procedurally generated using Python for each given neural network architecture.

Based on the complexity of each of these models, they can be  grouped with similar models in terms of mathematical properties (i.e. whether the same set of algorithms are valid), with the end goal being to generate a better understanding of the applicability of data dimensionality reduction algorithms based on the structure of the underlying dynamical systems.
