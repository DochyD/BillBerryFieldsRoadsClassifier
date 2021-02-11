
#Every imports we need to

from fastai.vision.widgets import *
from fastai.vision.all import *

dataset = Path("data")

# get a list of all filename in our dataset
fns = get_image_files(dataset)

# get number of images we got. if 0 there is an error with the loading of dataset.
print("Number of images in dataset : ",len(fns))


#We will augment our dataset by taking some randoms square (200x200) in the images
item_tfms = RandomResizedCrop(200, min_scale=1)
#And we will also flip the images (verticaly), change the brightness, and rotate the images, to create more data.
tfms = [Flip(p=1), Brightness(max_lighting=0.8, p=1, draw=None, batch=False), Rotate(max_deg=30, p=0.5)]

dataBlock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # tuple with ImageBlock and CategoryBlock
    get_items=get_image_files,  # use the get_image_files function
    splitter=RandomSplitter(valid_pct=0.2, seed=42),  # RandomSplitter
    get_y=parent_label,  # classes are separated into folders : use parent_label func
    item_tfms=item_tfms, 
    batch_tfms=tfms
    )

#We transform the datablock into a data loader
dls = dataBlock.dataloaders(dataset)


#Creation of a model, downloaded from the internet (resnet34), and get some informations such as error_rate and accuracy. They will show p when we launch ou program.
learner = cnn_learner(dls, resnet34, metrics=[error_rate,accuracy])
#We train our learner with 3 epochs
learner.fine_tune(3)
#we save the model if we want to train it again later.
learner.save(Path("Model"))





