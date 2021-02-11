# BillBerryFieldsRoadsClassifier

TODO :

pip3 install fastai

pip3 install ipywidgets

Script Should run on Linux With those packages. (It worked on an emulated Linux)

Then you can launch the scripts and compare the accuracies of both. (This should only take a few minutes)

There is only 3 epochs during the training since it's enough to obtain 100% accuracy with the data augmentation.

In order to create a validation set, I took 20% of the initial data, so 80% is used for training.

If you want to see some results (eg : with images etc...) you can go on this link to see the notebook I worked on.

https://nbviewer.jupyter.org/github/DochyD/BillBerryFieldsRoadsClassifier/blob/master/Classifier_ROADS_FIELDS.ipynb

You will first see the implementation with no data augmentation, the with data augmentation.

Later in the notebook, you will see some exemple of data augmentation I used. (Flip, Brightness, and rotate)

I could have used more data augmentation, but I choose to use the one I found the most useful. (And I did not wanted to overfit. I would have tried/ experiment more if I had more time)
