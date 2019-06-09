# NeurIPS 2019 - ID 6971 - Code Submission

Dear reviewers, 

we want to make it as easy as possible for you to run our code. The easiest way to run the experiments is via [myBinder](https://gke.mybinder.org/). This platform automatically builds an docker image, creates a conda environment and loads all the necessary packages for you. You then just have to run the Jupyter notebook ``` run_experiments.ipynb ```. There are two options available to use the myBinder platform:

* One-click solution: Just press on this badge [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neurips-2019-id-6971/submission/master?filepath=run_experiments.ipynb) and everything happens automatically.
* Almost one-click solution: Copy and paste the repository's URL (_https://github.com/neurips-2019-id-6971/submission_) in the respective field on the [myBinder](https://gke.mybinder.org/) homepage and press 'launch'.

Just in case something goes wrong with myBinder, here is another option to run the code:

* The more tedious path: Clone the repository on your local machine and create a conda environment with ```conda env create --file=environment.yaml```. Then run ```jupyter notebook ```. This opens a browser and you just need to execute the ``` run_experiments.ipynb ```. 


_Copyright (c) 2019, Copyright holder of the paper "Noisy-Input Entropy Search for Efficient Robust Bayesian Optimization" submitted to NeurIPS 2019 for review.
All rights reserved._
