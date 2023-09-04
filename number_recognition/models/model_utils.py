from multiprocessing import Process
import subprocess
import warnings
import os
import yaml
import number_recognition.config as cfg
import cv2
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD


def mount_nextcloud(
    nextCloudUser="DEEP_IAM-41539aaf-8a57-4a53-9707-26c77f635c69",
    mountHere="/home/petersi/davfs",
    sudo=True,
):
    command = [
        "mount",
        "-t",
        "davfs",
        "-o",
        "noexec",
        f"https://data-deep.a.incd.pt/remote.php/dav/files/{nextCloudUser}",
        f"{mountHere}",
    ]
    if sudo:
        command = ["sudo"] + command

    result = subprocess.Popen(
        command
    )  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


def umount_nextcloud(umountFrom="/home/petersi/davfs", sudo=True):
    command = ["umount", f"{umountFrom}"]
    if sudo:
        command = ["sudo"] + command

    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


def rclone_directory(fromPath, toPath):
    command = ["rclone", "copy", f"rshare:{fromPath}", f"{toPath}"]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


def check_path_existence(listOfPaths):
    notFound = []
    for path in listOfPaths:
        if os.path.exists(path) is False:
            notFound.append(path)
    print(f"Number of not found paths: {len(notFound)}")
    return notFound


def load_config_yaml(pathYaml, part=""):
    with open(pathYaml) as yamlFile:
        config = yaml.safe_load(yamlFile)
    if part == "":
        return config
    else:
        return config[part]


def read_data(dataPath, ydata=True):
    dataX = []
    dataY = []
    fileNames = []
    dirs = [
        name
        for name in os.listdir(dataPath)
        if os.path.isdir(os.path.join(dataPath, name))
    ]
    if len(dirs) == 0:
        dirs = ["."]
    for dr in dirs:  # range(0,10):
        print(f"dirs == {dr}")
        for fl in os.listdir(dataPath + "/" + dr):
            dataX.append(cv2.imread(dataPath + "/" + dr + "/" + fl))
            if ydata:
                dataY.append(int(dr))
            else:
                fileNames.append(fl)
    if ydata:
        return dataX, dataY
    else:
        return dataX, fileNames


def image_data_reshape(dataX):
    if isinstance(dataX, list):
        dataXOut = np.zeros(
            [len(dataX), np.shape(dataX[0])[0], np.shape(dataX[0])[1]], dtype=float
        )
        for i in range(len(dataX)):
            dataXOut[i, :, :] = dataX[i][:, :, 0] / 255
    else:
        dataXOut = np.zeros(
            [1, np.shape(dataX[0])[0], np.shape(dataX[0])[1]], dtype=float
        )
        dataXOut[0, :, :] = dataX[:, :, 0] / 255
    return dataXOut


def define_model(parameters):  # dotiahnut hodnoty z configu
    model = Sequential()
    parameters = parameters["model_parameters"]

    # neural network architecture
    nn_layer = parameters["nn_layer"]
    for i in range(len(nn_layer)):
        keys = nn_layer[i].keys()
        layerName, otherSettings = [], []
        s = list()
        for key in keys:
            if key == "name":
                layerName = nn_layer[i]["name"]
            elif key == "other_settings":
                otherSettings = [nn_layer[i]["other_settings"]]
            else:
                s.append(key + "=" + nn_layer[i][key])
        s = otherSettings + s
        s = ", ".join(s)
        s = layerName + "(" + s + ")"
        eval("model.add(" + s + ")")

    # optimizer settings
    opt = parameters["optimizer"]
    keys = opt.keys()
    optName, otherSettings = [], []
    s = list()
    for key in keys:
        if key == "name":
            optName = opt["name"]
        elif key == "other_settings":
            otherSettings = [opt["other_settings"]]
        else:
            s.append(key + "=" + opt[key])
    s = otherSettings + s
    s = ", ".join(s)
    s = optName + "(" + s + ")"
    optimizer = s

    # model compile
    mcompile = parameters["model_compile"]
    keys = mcompile.keys()
    otherSettings = []
    s = list()
    for key in keys:
        if key == "optimizer":
            if mcompile["optimizer"] != "":
                optimizer = mcompile["optimizer"]
            optimizer = "optimizer = " + optimizer
        elif key == "other_settings":
            otherSettings = [mcompile["other_settings"]]
        else:
            s.append(key + "=" + mcompile[key])
    s = otherSettings + s
    s = ", ".join(s)
    eval("model.compile(" + s + ")")
    return model


def train_model(dataX, dataY, parameters):
    n_folds = parameters["train_model_settings"]["n_folds"]
    epochs = parameters["train_model_settings"]["epochs"]
    batch_size = parameters["train_model_settings"]["batch_size"]
    scores, histories, models = list(), list(), list()

    kfold = KFold(n_folds, shuffle=True, random_state=1)  # prepare cross validation
    for train_ix, test_ix in kfold.split(dataX):  # enumerate splits
        model = define_model(parameters)  # define model
        trainX, trainY = dataX[train_ix, :, :], dataY[train_ix]  # select rows for train
        testX, testY = dataX[test_ix, :, :], dataY[test_ix]  # select rows for test
        history = model.fit(
            trainX,
            trainY,
            epochs=epochs,  # fit model
            batch_size=batch_size,
            validation_data=(testX, testY),
            verbose=0,
        )
        _, acc = model.evaluate(testX, testY, verbose=0)  # evaluate model
        print("> %.3f" % (acc * 100.0))
        scores.append(acc)  # stores scores
        histories.append(history)
        models.append(model)
    return scores, histories, models


def prediction_vector(model, dataX):
    prediction = model.predict(dataX)
    prediction = np.argmax(prediction, axis=1)
    return prediction


def test_model(model, dataX, dataY=[]):
    prediction = prediction_vector(model, dataX)
    if dataY == []:
        return prediction
    else:
        acc = np.sum(prediction == dataY) / len(dataY)
        print(prediction[0:10])
        print(dataY[0:10])
        return prediction, acc


def load_model(modelPath, parameters):
    model = define_model(parameters)
    model.load_weights(modelPath, skip_mismatch=False, by_name=False, options=None)
    return model


def save_model(model, modelPath):
    print(modelPath)
    model.save_weights(modelPath)
    return 1
