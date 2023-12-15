import json
import os
import random
import time
from datetime import datetime

import humanize
import numpy
import numpy as np
import pandas as pd
import psutil
import tensorflow
from PIL import Image

from sample_satellite import loader
from sample_satellite.datahandler import classify_file_path_by_date, get_weather_data, generate_images_from_date, \
    get_memory_usage, randomise_data_selection


def init_data(seed=0, dataset_use=50):
    npy_data_path = classify_file_path_by_date()
    weathers_data = get_weather_data()
    
    t = time.time()
    print("## Start v1")
    loader_v1 = loader.load_all_data_v1(seed, dataset_use, weathers_data, npy_data_path)
    print('## Loader v1 : ', round(float(time.time() - t), 2), 's')
    
    return loader_v2


if __name__ == "__main__":
    data_init_ = init_data(dataset_use=500)
    # date_use_ = data_init_['date_use']
    npy_data_path_ = classify_file_path_by_date()
    
    train_X, train_Y, test_X, test_Y = data_init_['dataset']
    
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Flatten(input_shape=(32,32)),
        tensorflow.keras.layers.Dense(128, activation='relu'),
        tensorflow.keras.layers.Dense(2)
    ])
    
    model.compile(optimizer="adam",
                  loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ["accuracy"]
                  )
    
    model.fit(train_X, train_Y, epochs=15)

