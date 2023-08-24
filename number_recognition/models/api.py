# -*- coding: utf-8 -*-
"""
Integrate a model with the DEEP APIm
"""

import os
import sys
import json
import argparse
import pkg_resources

# import project's config.py
import model_utils as mutils
from aiohttp.web import HTTPBadRequest

from functools import wraps

# Authorization
# from flaat import Flaat
# flaat = Flaat()

from tensorflow.keras.utils import to_categorical

base_directory = os.path.dirname(os.path.abspath(__file__))
base_directory = os.path.dirname(os.path.dirname(base_directory))
sys.path.append(base_directory)
sys.path.append(base_directory + "/number_recognition/models")
# next import cannot be at the begin of script due to
# missing paths - look commands sys.path.append(...)
import number_recognition.config as cfg

# sys.path.append(base_directory + "/flaat")


def _catch_error(f):
    """Decorate function to return an error as HTTPBadRequest, in case"""

    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)

    return wrap


def _fields_to_dict(fields_in):
    """
    Example function to convert mashmallow fields to dict()
    """
    dict_out = {}

    for key, val in fields_in.items():
        param = {}
        param["default"] = val.missing
        param["type"] = type(val.missing)
        if any(
            (
                key == "files",
                key == "urls",
                key == "model_path",
                key == "rclone_nextcloud",
            )
        ):
            param["type"] = str

        val_help = val.metadata["description"]
        if "enum" in val.metadata.keys():
            val_help = "{}. Choices: {}".format(val_help, val.metadata["enum"])
        param["help"] = val_help

        try:
            val_req = val.required
        except Exception:
            val_req = False
        param["required"] = val_req

        dict_out[key] = param
    return dict_out


def get_metadata():
    """
    Function to read metadata
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_metadata
    :return:
    """

    module = __name__.split(".", 1)

    try:
        pkg = pkg_resources.get_distribution(module[0])
    except pkg_resources.RequirementParseError:
        # if called from CLI, try to get pkg from the path
        distros = list(pkg_resources.find_distributions(cfg.BASE_DIR, only=True))
        if len(distros) == 1:
            pkg = distros[0]
    except Exception as e:
        raise HTTPBadRequest(reason=e)

    # One can include arguments for train() in the metadata
    train_args = _fields_to_dict(get_train_args())
    # make 'type' JSON serializable
    for key, val in train_args.items():
        train_args[key]["type"] = str(val["type"])

    # One can include arguments for predict() in the metadata
    predict_args = _fields_to_dict(get_predict_args())
    # make 'type' JSON serializable
    for key, val in predict_args.items():
        predict_args[key]["type"] = str(val["type"])

    meta = {
        "name": None,
        "version": None,
        "summary": None,
        "home-page": None,
        "author": None,
        "author-email": None,
        "license": None,
        "help-train": train_args,
        "help-predict": predict_args,
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower()  # to avoid inconsistency due to letter cases
        for par in meta:
            if line_low.startswith(par.lower() + ":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def warm():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.warm
    :return:
    """
    # e.g. prepare the data


def get_validate_args():
    return cfg.ValidationArgsSchema().fields


def get_predict_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    return cfg.PredictArgsSchema().fields


def set_kwargs(argument, **kwargs):
    if argument == "model_path":
        try:
            if not kwargs["model_path"] is None and kwargs["model_path"] != "":
                model_path = kwargs["model_path"]
            else:
                model_path = cfg.MODEL_PATH
        except Exception:
            model_path = cfg.MODEL_PATH
        return model_path

    elif argument == "rcloneNextcloud":
        rcloneNextcloud = False
        try:
            rcloneNextcloud = eval(kwargs["rclone_nextcloud"])
        except Exception:
            rcloneNextcloud = False
        return rcloneNextcloud

    else:
        return f"Bad 'variable' argument: {argument}"


@_catch_error
def validate(**kwargs):
    if not mutils.check_path_existence(cfg.CONFIG_YAML):
        print(f"Error: yaml config is missing, cfg.CONFIG_YAML == {cfg.CONFIG_YAML}")
    else:
        # load model parameters
        parameters = mutils.load_config_yaml(cfg.CONFIG_YAML)

    # model_path
    model_path = set_kwargs("model_path", **kwargs)

    # rclone_nextcloud
    rclone_nextcloud = set_kwargs("rclone_nextcloud", **kwargs)

    if rclone_nextcloud:
        mutils.rclone_directory(cfg.NEXTCLOUD_VALIDATION_DIR, cfg.VALIDATION_DIR)
        if not mutils.check_path_existence(cfg.VALIDATION_DIR):
            print(f"Warning: rclone Nextclone validation dataset was not successful.")

    if not mutils.check_path_existence(cfg.VALIDATION_DIR):
        print(
            f"Error:validation data is missing,cfg.VALIDATION_DIR=={cfg.VALIDATION_DIR}"
        )
    else:
        # nacitanie testovacich
        dataValX, fileNames = mutils.read_data(cfg.VALIDATION_DIR, ydata=False)
        dataValX = mutils.image_data_reshape(dataValX)
        # nacitaj model
        modelLoad = mutils.load_model(model_path, parameters)
        # testovanie modelu
        prediction = mutils.test_model(modelLoad, dataValX)
        print("Predictions:")
        print(fileNames)
        for i in range(len(fileNames)):
            print(f"{fileNames[i]}: {prediction[i]}")


@_catch_error
def predict(**kwargs):
    """
    Function to execute prediction
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.predict
    :param kwargs:
    :return:
    """

    # if (not any([kwargs['urls'], kwargs['files']]) or
    #         all([kwargs['urls'], kwargs['files']])):
    #     raise Exception("You must provide either 'url' or 'data' in the payload")

    # if kwargs['files']:
    #     kwargs['files'] = [kwargs['files']]  # patch until list is available
    #     return _predict_data(kwargs)
    # elif kwargs['urls']:
    #     kwargs['urls'] = [kwargs['urls']]  # patch until list is available
    #     return _predict_url(kwargs)

    if not mutils.check_path_existence(cfg.CONFIG_YAML):
        print(f"Error: yaml config is missing, cfg.CONFIG_YAML == {cfg.CONFIG_YAML}")
    else:
        # load model parameters
        parameters = mutils.load_config_yaml(cfg.CONFIG_YAML)

    # model_path
    model_path = set_kwargs("model_path", **kwargs)

    # rclone_nextcloud
    rclone_nextcloud = set_kwargs("rclone_nextcloud", **kwargs)

    if rclone_nextcloud:
        mutils.rclone_directory(cfg.NEXTCLOUD_TEST_DIR, cfg.TEST_DIR)
        if not mutils.check_path_existence(cfg.TEST_DIR):
            print(f"Warning: rclone Nextclone test dataset was not successful.")

    if not mutils.check_path_existence(cfg.TEST_DIR):
        print(f"Error: testing data is missing, cfg.TEST_DIR == {cfg.TEST_DIR}")
    else:
        # nacitanie testovacich
        dataTestX, dataTestY = mutils.read_data(cfg.TEST_DIR)
        dataTestX = mutils.image_data_reshape(dataTestX)
        # nacitaj model
        modelLoad = mutils.load_model(model_path, parameters)
        # testovanie modelu
        prediction, acc = mutils.test_model(modelLoad, dataTestX, dataTestY)
        print(f"Predictions: {prediction}")
        print(f"Model accuracy: {acc}")


def _predict_data(*args):
    """
    (Optional) Helper function to make prediction on an uploaded file
    """
    message = "Not implemented (predict_data())"
    message = {"Error": message}
    return message


def _predict_url(*args):
    """
    (Optional) Helper function to make prediction on an URL
    """
    message = "Not implemented (predict_url())"
    message = {"Error": message}
    return message


def get_train_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    :param kwargs:
    :return:
    """
    return cfg.TrainArgsSchema().fields


###
# @flaat.login_required() line is to limit access for only authorized people
# Comment this line, if you open training for everybody
# More info: see https://github.com/indigo-dc/flaat
###
# @flaat.login_required() # Allows only authorized people to train
def train(**kwargs):
    """
    Train network
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.train
    :param kwargs:
    :return:
    """

    message = {
        "status": "ok",
        "training": [],
    }

    # use the schema
    schema = cfg.TrainArgsSchema()
    # deserialize key-word arguments
    train_args = schema.load(kwargs)
    print(f"train_args['model_path'] == {train_args['model_path']}")
    print(f"train_args['rclone_nextcloud'] == {train_args['rclone_nextcloud']}")

    # 1. implement your training here
    # 2. update "message"

    train_results = {"Error": "No model implemented for training (train())"}
    message["training"].append(train_results)

    # model_path
    model_path = set_kwargs("model_path", **kwargs)

    # rclone_nextcloud
    rclone_nextcloud = set_kwargs("rclone_nextcloud", **kwargs)

    if rclone_nextcloud:
        mutils.rclone_directory(cfg.NEXTCLOUD_TRAIN_DIR, cfg.TRAIN_DIR)
        if not mutils.check_path_existence(cfg.TRAIN_DIR):
            print(f"Warning: rclone Nextclone train dataset was not successful.")

    # CONFIG.yaml
    if not mutils.check_path_existence(cfg.CONFIG_YAML):
        print(f"Error: yaml config is missing, cfg.CONFIG_YAML == {cfg.CONFIG_YAML}")
    else:
        # load model parameters
        parameters = mutils.load_config_yaml(cfg.CONFIG_YAML)

    if not mutils.check_path_existence(cfg.TRAIN_DIR):
        print(f"Error: training data is missing, cfg.TRAIN_DIR == {cfg.TRAIN_DIR}")
        return message

    # load train data
    dataTrainX, dataTrainY = mutils.read_data(cfg.TRAIN_DIR)
    dataTrainX = mutils.image_data_reshape(dataTrainX)
    # model training
    trainScores, trainHistories, trainModels = mutils.train_model(
        dataTrainX, to_categorical(dataTrainY), parameters
    )
    # model saving
    mutils.save_model(trainModels[1], model_path)

    return message


# during development it might be practical
# to check your code from CLI (command line interface)
def main():
    """
    Runs above-described methods from CLI
    (see below an example)
    """

    if args.method == "get_metadata":
        meta = get_metadata()
        print(json.dumps(meta))
        return meta
    elif args.method == "validate":
        # [!] you may need to take special care in the case of args.files [!]
        results = validate(**vars(args))
        print(json.dumps(results))
        return results
    elif args.method == "predict":
        # [!] you may need to take special care in the case of args.files [!]
        results = predict(**vars(args))
        print(json.dumps(results))
        return results
    elif args.method == "train":
        results = train(**vars(args))
        print(json.dumps(results))
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model parameters", add_help=False)

    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
        help='methods. Use "api.py method --help" to get more info', dest="method"
    )

    # -------------------------------------------------------------------------------------
    # configure parser to call get_metadata()
    get_metadata_parser = subparsers.add_parser(
        "get_metadata", help="get_metadata method", parents=[parser]
    )
    # normally there are no arguments to configure for get_metadata()

    # -------------------------------------------------------------------------------------
    # configure arguments for validate()
    validate_parser = subparsers.add_parser(
        "validate", help="commands for validation", parents=[parser]
    )
    # one should convert get_validate_args() to add them in validation_parser
    # For example:
    validation_args = _fields_to_dict(get_validate_args())
    for key, val in validation_args.items():
        validate_parser.add_argument(
            "--%s" % key,
            default=val["default"],
            type=val["type"],
            help=val["help"],
            required=val["required"],
        )

    # -------------------------------------------------------------------------------------

    # configure arguments for predict()
    predict_parser = subparsers.add_parser(
        "predict", help="commands for prediction", parents=[parser]
    )
    # one should convert get_predict_args() to add them in predict_parser
    # For example:
    predict_args = _fields_to_dict(get_predict_args())
    for key, val in predict_args.items():
        predict_parser.add_argument(
            "--%s" % key,
            default=val["default"],
            type=val["type"],
            help=val["help"],
            required=val["required"],
        )

    # -------------------------------------------------------------------------------------
    # configure arguments for train()
    train_parser = subparsers.add_parser(
        "train", help="commands for training", parents=[parser]
    )
    # one should convert get_train_args() to add them in train_parser
    # For example:
    train_args = _fields_to_dict(get_train_args())
    for key, val in train_args.items():
        train_parser.add_argument(
            "--%s" % key,
            default=val["default"],
            type=val["type"],
            help=val["help"],
            required=val["required"],
        )

    args = cmd_parser.parse_args()

    main()
