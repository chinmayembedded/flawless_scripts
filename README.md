# flawless_scripts

## Main tasks

Task 1: Done


Task 2: Done. Create a function in data_split for splitting train and validation sets


Task 3: Done. Random noise addition function in noise.py


Task 4: Done.

Kernel size 5, Strides 2 and padding 0 is used. As the model has only two layers for 28x28 image, kernel size is kept as moderate such as 5 in this experiments.
It is a digit classification problem which requires to learn only generic features rather than intricate or detailed one. Hence filter size is 5 and strides are 2 where, model tolerates the loss of exact image.


Task 5: Partial

A softmax function is created but it does not work perfectly and shows negative training loss

Task 6: Done


Task 7: Partial.

Model accuracy is calculated on test dataset



Based on:
https://discuss.pytorch.org/t/custom-softmax-function/5048
https://debuggercafe.com/adding-noise-to-image-data-for-deep-learning-data-augmentation/
https://nextjournal.com/gkoehler/pytorch-mnist