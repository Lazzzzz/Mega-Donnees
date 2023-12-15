import time
from multiprocessing import Queue, Process

from sample_satellite.datahandler import generate_images_from_date, get_memory_usage, randomise_data_selection


def load_all_data_v1(seed, dataset_use, weathers_data, npy_data_path ):
    loaded_data = [[], []]
    
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
                
    return {'date_use' : date_used, 'dataset' : randomise_data_selection(seed, loaded_data)}

def split_list(lst, parts):
    sublist_size = len(lst) // parts
    remainder = len(lst) % parts

    start = 0
    result = []
    
    for i in range(parts):
        if i < remainder:
            end = start + sublist_size + 1
        else:
            end = start + sublist_size

        result.append(lst[start:end])
        start = end

    return result
