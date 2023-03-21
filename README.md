# DSM Modeler

This software can produce LoD 2 3D models of buildings from Digital Surface Model (DSM) data using as input:
- DSM raster data (georeferenced ASC file)
- Shapefile (GeoJSON) of the building footprint

A full description of the algorithm can be found in 

L. Adreani, P. Bellini, C. Colombo, M. Fanfani, P. Nesi, G. Pantaleo, R. Pisanu, "Integrated Digital Twin for Smart City Solutions", Multimedia Tools and Applications (under review)

## Usage
1. Download the code;
2. Install the required Python modules using the `requirements.txt`
2. Download the `canny_edge_detector.py` module from [here](https://github.com/FienSoP/canny_edge_detector), and put it into the function folder;
3. Run `main.py`

Note: during the execution several figures are displayed. To select which figure to show, use the boolean variables in the `get_3D_model.py` file.

## Data
The example data in the data folder are provided only to demonstrate the functionality of the software.
- the building.geojson shape file was extracted from [OpenStreetMap](https://www.openstreetmap.org/)
- the dsm_example.asc and the dtm_example.asc were clipped from DSM and DTM data provided by Sistema Informativo Territoriale ed Ambientale (GIS local system) of Tuscany Region