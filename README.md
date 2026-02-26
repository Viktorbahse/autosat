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

#### **Подготовка и активация окружения**
```bash
    # setup uv environment
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create a .venv local virtual environment (if it doesn't exist)
    [ -d ".venv" ] || uv venv

    # install requirements + pre-commit hook
    make setup

    # activate environment
    source .venv/bin/activate
```

#### **DVC pipeline**

```bash
    # запустить весь pipeline
    dvc repro
```

```bash
    # запустить pipeline по шагам
    
    # stage 1:
    # скачать c https://download.geofabrik.de
    # OSM разметку для большой карты региона
    # и сохранить разметку для prepare_data.map_box 
    dvc repro download_and_extract_osm

    # stage 2:
    # скачать и сохранить изображения и маски
    dvc repro prepare_satellite_data
```

#### **Pre-commit check**

```bash
    make pre-commit-check
```


<!-- uv pip install --find-links https://girder.github.io/large_image_wheels gdal  pyproj -->