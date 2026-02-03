import argparse
import concurrent.futures
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from shapely.geometry import Polygon, box


def parse_coordinates() -> Tuple[float, float, float, float]:
    
    parser = argparse.ArgumentParser(
        description='Извлечение OSM данных для указанного региона'
    )
    
    parser.add_argument(
        '--bounds',
        type=float,
        nargs=4,
        required=True,
        metavar=('MIN_LON', 'MIN_LAT', 'MAX_LON', 'MAX_LAT'),
        help='Границы прямоугольника в формате: min_lon min_lat max_lon max_lat'
    )
    
    parser.add_argument(
        '--example',
        action='store_true',
        help='Показать пример использования'
    )
    
    args = parser.parse_args()
    
    if args.example:
        print("Пример использования:")
        print("python script.py --bounds 29.97070 59.32198 33.94775 61.32827")
        print("python script.py --bounds 37.0 55.0 38.0 56.0")
        sys.exit(0)
    
    min_lon, min_lat, max_lon, max_lat = args.bounds
    
    
    
    return (min_lon, min_lat, max_lon, max_lat)

def find_ring_containing_rectangle(
    geometry_data: Union[List, Tuple],
    rect_bounds: Tuple[float, float, float, float]
) -> bool:
    """
    Проверяет, содержит ли какое-либо кольцо в геометрическом массиве полностью прямоугольник.
    
    Параметры:
    -----------
    geometry_data : list/tuple
        Многомерный массив геометрических данных в формате:
        [полигоны][кольца][точки][долгота, широта]
        
    rect_bounds : tuple
        Границы прямоугольника в формате (min_lon, min_lat, max_lon, max_lat)
        
        
    Возвращает:
    -----------
    bool
        True, если найдется кольцо, полностью содержащее прямоугольник
    """
    
    rect = box(*rect_bounds)
    
    def extract_rings(data, rings_list=None, level=0):
        if rings_list is None:
            rings_list = []
        
        if (isinstance(data, (list, tuple)) and len(data) > 0 and
            isinstance(data[0], (int, float))):
            return data
        
        if isinstance(data, (list, tuple)):
            if (len(data) > 0 and isinstance(data[0], (list, tuple)) and
                len(data[0]) > 0 and isinstance(data[0][0], (int, float))):
                rings_list.append(data)
            else:
                for item in data:
                    extract_rings(item, rings_list, level + 1)
        
        return rings_list
    
    all_rings = extract_rings(geometry_data)
    
    for ring_coords in all_rings:
        try:
            if len(ring_coords) >= 3:
                first_point = ring_coords[0]
                last_point = ring_coords[-1]
                
                if first_point != last_point:
                    closed_ring = ring_coords + [first_point]
                else:
                    closed_ring = ring_coords
                
                ring_polygon = Polygon(closed_ring)

                contains = ring_polygon.covers(rect)
                
                if contains:
                    return True
                    
        except Exception as e:
            print(f"Ошибка при обработке вхождения области в регион: {e}")
            continue
    
    return False


def find_matching_features(
    geojson_data: Dict[str, Any],
    rect_bounds: Tuple[float, float, float, float],
) -> List[Dict[str, Any]]:
    """
    Проходит по GeoJSON FeatureCollection и возвращает все features,
    для которых find_ring_containing_rectangle возвращает True.
    
    Параметры:
    -----------
    geojson_data : dict
        GeoJSON объект типа FeatureCollection
    rect_bounds : tuple
        Границы прямоугольника (min_lon, min_lat, max_lon, max_lat)
        
    Возвращает:
    -----------
    List[Dict]
        Список features, которые содержат прямоугольник
    """
    
    if geojson_data.get('type') != 'FeatureCollection':
        raise ValueError("Ожидается GeoJSON FeatureCollection")
    
    features = geojson_data.get('features', [])
    matching_features = []
    
    for feature in features:
        geometry = feature.get('geometry', {})
        
        if not geometry or 'coordinates' not in geometry:
            continue  
        
        contains = find_ring_containing_rectangle(
            geometry_data=geometry.get('coordinates', []),
            rect_bounds=rect_bounds
        )
        
        if contains:
            matching_features.append(feature)
    
    return matching_features


def select_smallest_pbf_feature(
    matching_features: List[Dict[str, Any]],
    timeout: int = 10,
    max_workers: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Выбирает feature с наименьшим по размеру PBF файлом.
    
    Параметры:
    -----------
    matching_features : List[Dict]
        Список features, полученных из find_matching_features
    timeout : int
        Таймаут для HTTP-запросов в секундах
    max_workers : int
        Максимальное количество потоков для параллельных запросов
        
    Возвращает:
    -----------
    Optional[Dict]
        Feature с наименьшим PBF файлом или None, если не удалось определить
    """
    
    if not matching_features:
        return None
    
    if len(matching_features) == 1:
        return matching_features[0]
    
    def get_file_size(url: str) -> Optional[int]:
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            if response.status_code == 200:
                size_header = response.headers.get('Content-Length')
                if size_header:
                    return int(size_header)
                
                response = requests.get(
                    url, 
                    timeout=timeout, 
                    headers={'Range': 'bytes=0-0'},
                    stream=True
                )
                if response.status_code == 206:
                    content_range = response.headers.get('Content-Range')
                    if content_range:
                        total_size = content_range.split('/')[-1]
                        if total_size.isdigit():
                            return int(total_size)
            return None
        except Exception as e:
            print(f"Ошибка при получении размера {url}: {e}")
            return None
    
    features_with_size = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feature = {}
        for feature in matching_features:
            urls = feature.get('properties', {}).get('urls', {})
            pbf_url = urls.get('pbf')
            if pbf_url:
                future = executor.submit(get_file_size, pbf_url)
                future_to_feature[future] = feature
        
        for future in concurrent.futures.as_completed(future_to_feature):
            feature = future_to_feature[future]
            try:
                file_size = future.result(timeout=timeout)
                features_with_size.append((feature, file_size))
                
                props = feature.get('properties', {})
                size_str = f"{file_size / (1024*1024):.1f} MB" if file_size else "не определен"
                print(f"  {props.get('name')}: {size_str}")
                
            except Exception as e:
                print(f"Ошибка для {feature.get('properties', {}).get('name')}: {e}")
                features_with_size.append((feature, None))
    
    valid_features = [(f, size) for f, size in features_with_size if size is not None]
    
    if not valid_features:
        print("Не удалось определить размер ни одного файла. Возвращаем первый feature.")
        return matching_features[0]
    
    smallest_feature, smallest_size = min(valid_features, key=lambda x: x[1])
    
    props = smallest_feature.get('properties', {})
    print(f"\nНаименьший файл: {props.get('name')} ({smallest_size / (1024*1024):.1f} MB)")
    
    return smallest_feature

if __name__ == "__main__":
    try:
        rect_bounds = parse_coordinates()
        print(f"Используются координаты: {rect_bounds}")
        
        url = "https://download.geofabrik.de/index-v1.json"

        response = requests.get(url)
        response.raise_for_status()  
        geojson_data = response.json()
        
        matching = find_matching_features(
            geojson_data=geojson_data,
            rect_bounds=rect_bounds
        )
        if len(matching)==0:
            print('Регион содержащий выбранную область не найден.')
        else:
            smalest = select_smallest_pbf_feature(matching)
            os.system(f"wget --limit-rate=10M -P data/osm-data/ {smalest['properties']['urls']['pbf']}")
        
    except ValueError as e:
        print(f"Ошибка в координатах: {e}")
        sys.exit(1)

    
    
