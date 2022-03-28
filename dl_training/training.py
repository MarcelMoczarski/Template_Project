from dl_framework import data, callbacks, learner, model, loss_functions
from pathlib import Path

mnt_dir = Path("/content")


val_split = 0.2
batch_size = 64
num_classes = 10

x_train, y_train, x_test, y_test = data.get_dataset("torchvision", "MNIST", mnt_dir)
x_train, y_train, x_valid, y_valid = data.split_trainset(x_train, y_train, val_split)

train_ds = data.Dataset(x_train, y_train)
valid_ds = data.Dataset(x_valid, y_valid)
test_ds = data.Dataset(x_test, y_test)

train_dl, valid_dl, test_dl = data.get_dls(train_ds, valid_ds, test_ds, batch_size)
train_db = data.DataBunch(train_dl, valid_dl, num_classes)

monitor = callbacks.Monitor_Cb(["valid_acc", "valid_loss", "loss"])
earlystopping = callbacks.EarlyStopping_Cb(monitor="valid_loss", patience=10)

callbackhandler = callbacks.CallbackHandler([monitor, earlystopping])

learn = learner.Learner(*model.get_model(train_db, model.Model_1), 
                           loss_functions.cross_entropy, 
                           train_db,
                           callbackhandler
                           ) 
learn.fit(20)