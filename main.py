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

def classify_file_path_by_date():
    classify_file = {}
    
    for root, dirs, files in os.walk(r'D:\MD - data\data'):
        for file in files:
            file = file.replace('.npy', '')
            try:
                date = datetime.strptime(file, '%Y%m%d%H')
                formatted_date = date.strftime('%Y-%m-%d-%H')
                
                classify_file[formatted_date] = os.path.join(root, file + '.npy')
            
            except Exception as e:
                pass
    
    sorted_data = {k: v for k, v in sorted(classify_file.items(), key=lambda item: datetime.strptime(item[0], '%Y-%m-%d-%H'))}
    
    return sorted_data

# Wip
def save_image(date, index, image_data):
    def convert_brightness_to_rgb(pixel_brightness):
        return pixel_brightness, pixel_brightness, pixel_brightness
    
    image = Image.new("RGB", (32, 32))
    
    for y in range(32):
        for x in range(32):
            brightness = image_data[y][x]
            rgb_value = convert_brightness_to_rgb(int(brightness * 255.0))
            image.putpixel((x, y), rgb_value)

    root_path = r'D:\MD - data\code\sample_satellite\results'
    path = os.path.join(root_path, date)

    if not os.path.exists(path):
        os.makedirs(path)
    
    image.save(os.path.join(path, f"{index}.png"))

def pixel_processing(rgb):
    brightness = sum(rgb) // 3
    return max(0, min(brightness / 300.0, 1))

def generate_image(data):
    image = np.zeros((32, 32), dtype=np.float64)
    all_nan = True
   
    for x_index, data2 in enumerate(data):
        for y_index, data3 in enumerate(data2):
            pixel_data = pixel_processing(data3[4:7])

            if pixel_data == pixel_data and pixel_data != 0:
                all_nan = False
            
            image[y_index, x_index] = pixel_data
            
            
    return image if not all_nan else None
    
def generate_images_from_date(date, data):
    data_ = numpy.load(data[date])
    quart_heure = data_[0]
    
    images = []
    
    for station_index, data_station in enumerate(quart_heure):
        array_2d = generate_image(data_station)
        if array_2d is not None:
            image_data = array_2d.reshape((32, 32)).tolist()
            images.append(image_data)
            # save_image(date, station_index, image_data)
        else:
            images.append(None)

    return images
    
def get_metadata():
    df = pd.read_csv('metadata.csv')[['station_code', 'name', 'country', 'latitude', 'longitude', 'savanna', 'GMT+']]
    return df[(df['latitude'] > 6) & (df['latitude'] < 15) & (df['longitude'] > -16)].sort_values(by='station_code')

def get_weather_data():
    metadata = get_metadata()
    weather_data_raw = pd.read_csv('weather_stations_combined_at_GMT.csv')

    correct_station_code = metadata['station_code'].values
    station_to_remove = [item for item in list(weather_data_raw)[2:] if item not in correct_station_code]
    
    weather_data_treated = weather_data_raw.drop(columns=station_to_remove)
    
    weather_data = {}
    
    for data in weather_data_treated.values:
        date = datetime.strptime(data[1], '%Y-%m-%d %H:%M:%S')
        formatted_date = date.strftime('%Y-%m-%d-%H')
        
        if formatted_date not in weather_data:
            weather_data[formatted_date] = []
            
        for station_index, station_code in enumerate(correct_station_code):
            weather_data[formatted_date].append(data[2 + station_index])
    
    return weather_data
    
def get_memory_usage():
    process = psutil.Process()
    memory_bytes = process.memory_info().rss
    
    return humanize.naturalsize(memory_bytes)

def randomise_data_selection(seed, loaded_data):
    rand = random.Random(seed)
    
    test_images = []
    test_labels = []
    
    for _ in range(int(len(loaded_data[0]) * 0.2)):
        index_to_remove = rand.randint(0, len(loaded_data[0]) - 1)
        test_images.append(loaded_data[0][index_to_remove])
        test_labels.append(loaded_data[1][index_to_remove])
        
        del loaded_data[0][index_to_remove]
        del loaded_data[1][index_to_remove]
    
    print(f'Train size : {len(loaded_data[0])}')
    print(f'Test size : {len(test_images)}')
    print(f'Total size : {len(loaded_data[0]) + len(test_images)}')
    
    return loaded_data[0], loaded_data[1], test_images, test_labels

def init_data(seed=0,dataset_use=50):
    loaded_data = [[], []]
    
    npy_data_path = classify_file_path_by_date()
    weathers_data = get_weather_data()
    
    data_use = 0
    date_used = []
    
    
    for date in list(npy_data_path.keys()):
        if data_use >= dataset_use:
            break
            
        if date in weathers_data:
            images = generate_images_from_date(date, npy_data_path)
            added = False
            for station_id, station_value in enumerate(weathers_data[date]):
                if station_value == station_value and images[station_id] is not None:
                    loaded_data[0].append(images[station_id])
                    loaded_data[1].append(0 if station_value < 0.1 else 1)
                    added = True
            if added:
                date_used.append(date)
                data_use += 1
            
                print(f"data loaded {round((data_use / dataset_use) * 100.0, 2)} %".ljust(20),
                  f"| Memory {get_memory_usage()}")
    
    return {'date_use' :date_used, 'dataset' : randomise_data_selection(seed, loaded_data)}

data_init_ = init_data(dataset_use=20)
date_use_ = data_init_['date_use']
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

