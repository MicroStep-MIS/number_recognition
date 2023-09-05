# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields
from marshmallow import Schema, INCLUDE
from datetime import datetime

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if "APP_INPUT_OUTPUT_BASE_DIR" in os.environ:
    env_in_out_base_dir = os.environ["APP_INPUT_OUTPUT_BASE_DIR"]
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = '[WARNING] "APP_INPUT_OUTPUT_BASE_DIR='
        msg = msg + '{}" is not a valid directory! '.format(env_in_out_base_dir)
        msg = msg + 'Using "BASE_DIR={}" instead.'.format(BASE_DIR)
        print(msg)

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VALIDATION_DIR = os.path.join(DATA_DIR, "validate")
MODELS_DIR = os.path.join(IN_OUT_BASE_DIR, "models")
MODEL_PATH = os.path.join(
    BASE_DIR, "models" "/model_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".h5"
)
CONFIG_YAML = os.path.join(BASE_DIR, "CONFIG.yaml")

NEXTCLOUD_DATA_DIR = "Mnist/data"
NEXTCLOUD_TRAIN_DIR = os.path.join(NEXTCLOUD_DATA_DIR, "train")
NEXTCLOUD_TEST_DIR = os.path.join(NEXTCLOUD_DATA_DIR, "test")
NEXTCLOUD_VALIDATION_DIR = os.path.join(NEXTCLOUD_DATA_DIR, "validate")


# Input parameters for predict() (deepaas>=1.0.0)
class ValidationArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields:
    # https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction

    model_path = fields.String(
        required=False,
        missing=None,
        description="Select a hdf5 model for the validation",
    )

    rclone_nextcloud = fields.String(
        required=False, missing=None, description="True/False for rclone copy nextcloud"
    )

    # an input parameter for validation
    arg1 = fields.Integer(
        required=False, missing=1, description="Input argument 1 for the validation"
    )


# Input parameters for predict() (deepaas>=1.0.0)
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields:
    # https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction
    files = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="data",
        location="form",
        description="Select a file for the prediction",
    )

    # to be able to provide an URL for prediction
    urls = fields.Url(
        required=False,
        missing=None,
        description="Provide an URL of the data for the prediction",
    )

    model_path = fields.String(
        required=False,
        missing=None,
        description="Select a hdf5 model for the prediction",
    )

    rclone_nextcloud = fields.String(
        required=False, missing=None, description="True/False for rclone copy nextcloud"
    )

    # an input parameter for prediction
    arg1 = fields.Integer(
        required=False, missing=1, description="Input argument 1 for the prediction"
    )


# Input parameters for train() (deepaas>=1.0.0)
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    model_path = fields.String(
        required=False, missing=None, description="Select name for a hdf5 model"
    )

    rclone_nextcloud = fields.String(
        required=False, missing=None, description="True/False for rclone copy nextcloud"
    )

    # available fields are e.g. fields.Integer(), fields.Str(), fields.Boolean()
    # full list of fields:
    # https://marshmallow.readthedocs.io/en/stable/api_reference.html
    arg1 = fields.Integer(
        required=False, missing=1, description="Input argument 1 for training"
    )
