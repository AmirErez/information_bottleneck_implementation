
Single Cell Data Compression
============================
An implementation of Chigirev and Bialek 2003

Purpose
-------

This repository is a simple implementation of a data compression algorithm first established by Chigirev and Bialek in NIPs (2003).  I find this method valuable for its ability to to find optimal represenations of data, even data that are highly nonlinear.  An interesting application is in finding the opitmal representation of data consisting of mixtures of populations.  In this scenario, the algorithm compresses the data by finding points that lie close to the respective modes of the respective mixtures.

In studying biological phenomena, individuals are often tasked to find subpopulations of individual cells in high dimensional data.  This task can be challenging due to the endogenous variability of molecules (e.g. protein) among single cell clones.  Furthermore, the task is becoming more challenging with technological advances such as Mass Cyotmetry and Single cell sequencing, in which measurements of single cells ranges from fifty to thousands of dimensions.

In this example, I present simulated data that highlights the aforementioned application.  I use these simulations to show how data compression provides a means to find populations of cells with high sensitivity.  Lastly, I use this as an example for people interested in applying this technique to their own real data.

Example, Identifying mixtures of Gaussian Distributions
-------------------------------------------------------
![histograms](figs/bimodal_gauss_sampling.pdf)
