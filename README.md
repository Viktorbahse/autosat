<h1 align="center">AutoSat</h1>

<p align="center">
  <strong>Бахурин Виктор Владимирович</strong><br/>
  End-to-end pipeline for feature extraction from aerial and satellite imagery.
</p>

<p align="center">
  <img src="assets/mask.png" alt="AutoSat: buildings segmentation" title="AutoSat pipeline extracting buildings" width="600"/>
  <br/>
  <i>Aerial imagery — segmentation mask</i>
</p>


## Table of Contents

1. [Usage](#usage)
    - [osm_data_extract](#osm_data_extract)
    - [extract](#extract)
    - [cover](#cover)
    - [download](#download)
    - [rasterize](#rasterize)

### osm_data_extract

Скачивает map.osm.pbf c сайта [Geofabrik](https://download.geofabrik.de) содержащий область введенную пользователем. Область задается координатами нижнего левого и верхнего правого углов прямоугольника. Результат выполнения file.osm.pbf.

Пример использования:

```bash
python osm_data_extract.py --bounds 29.97070 59.32198 33.94775 61.32827
```

### extract
Собирает GeoJSON информацию об объектах пяти возможных классов: "parking", "building", "road", "water", "field" чтобы содать на их основе тренировочный датасет. Результат выполнения файл/папка с файлами расширения .geojso.

Пример использования:

```bash
python extract.py --type water data/osm-data/map.osm.pbf 

python extract.py --type all data/osm-data/map.osm.pbf 
```

### cover

Создает csv файл, содиржащий список тайлов, которые покрывают все объекты полученные на шаге extract.
Результат .csv файл c (x, y, z).

Пример использования:

```bash
python cover.py --input data/road/roads-2b2bf4e09fb047ab83f70d355ce83659.geojson

python cover.py --type parking --zoom 18

python cover.py --type all --zoom 16
```

### download

Инструмент для пакетной загрузки тайлов по списку. Реализована поддержка 4 стандартных провайдеров (Google, Bing, ArcGIS, Yandex) и возможность передачи произвольного URL от пользователя. Поддержка режима "all" для скачивания с всех ресурсов. Результат с директория с изображениями: path/z/x/y.jpg. 

Пример использования:

```bash
python download.py --input "data/building_tiles_16.csv" --url "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}" --out "data/tiles/google"

python download.py --input "data/building_tiles_16.csv" --source all

python download.py --input "data/building_tiles_16.csv" --source yandex
```

### rasterize

Создает бинарные маскии на основе GeoJSON информацию об объектах. Результат директория с изображениями-масками: path/z/x/y.png.

Пример использования:

```bash
python rasterize.py --zoom 16 --features 'data/building/' --tiles data/building_tiles_16.csv

```
