
#Every imports we need to
from fastai.vision.widgets import *
from fastai.vision.all import *

#Path to our data
dataset = Path("data")

# get a list of all filename in our dataset
fns = get_image_files(dataset)

# get number of images we got. if 0 there is an error with the loading of dataset.
print("Number of images in dataset : ",len(fns))

#The datablock only works with square images
item_tfms = [Resize(400)]
dataBlock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),  # tuple with ImageBlock and CategoryBlock (we store image associated with a category, for classification)
    get_items=get_image_files,  # use the get_image_files function
    splitter=RandomSplitter(valid_pct=0.2, seed=42),  # RandomSplitter, here we take 20% of our dataset to transform it into a validation set ( seed is used to get the same "random result every time, can be useful while experimenting")
    get_y=parent_label,  # classes are separated into folders : use parent_label func
    item_tfms=item_tfms# to transform the images, for exemple, wwe need to put it in a square format to put it in the dataloader
    )


#We turn our datablock into a dataloaders once the tranformations are done (eg : rezise, change color...)
dls = dataBlock.dataloaders(dataset)

#Creation of a model, downloaded from the internet (resnet34), and get some informations such as error_rate and accuracy. They will show p when we launch ou program
learner = cnn_learner(dls, resnet34, metrics=[error_rate,accuracy])
#We train our learner with 3 epochs
learner.fine_tune(3)
#we save the model if we want to train it again later.
learner.save(Path("Model"))





