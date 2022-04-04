import core.dl_framework as fw
import click
import toml
#test
# val_split = 0.2
# batch_size = 64
# num_classes = 10

# x_train, y_train, x_test, y_test = fw.data.get_dataset("torchvision", "MNIST", mnt_dir)
# x_train, y_train, x_valid, y_valid = fw.data.split_trainset(x_train, y_train, val_split)

# train_ds = fw.data.Dataset(x_train, y_train)
# valid_ds = fw.data.Dataset(x_valid, y_valid)
# test_ds = fw.data.Dataset(x_test, y_test)

# train_dl, valid_dl, test_dl = fw.data.get_dls(train_ds, valid_ds, test_ds, batch_size)
# train_db = fw.data.DataBunch(train_dl, valid_dl, num_classes)

# monitor = fw.callbacks.Monitor_Cb(["valid_acc", "valid_loss", "loss"])
# earlystopping = fw.callbacks.EarlyStopping_Cb(monitor="valid_loss", patience=2)

# callbacks = fw.callbacks.CallbackHandler([monitor, earlystopping])

# learn = fw.learner.Learner(*fw.model.get_model(train_db, fw.model.Model_2), 
#                            fw.loss_functions.cross_entropy, 
#                            train_db,
#                            callbacks
#                            ) 
# learn.fit(20)
def main()

if __name__ = "__main__":
    main()



#calc runtime
#average runtime
#toml file
#add paths in setup for start training
#add mechanism to safe model to gdrive. with option to resume training
#add save best model
#add tensorboard support
#add telegramlogger
#check pytest 
