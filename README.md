# WebGPU Viewsheds
- Load and render LAZ files.
- Sort points in morton order.
- Generate delauney mesh.
- Generate quadtree for raycasting acceleration.
- Raycast against nodes.
- Sort hit nodes per ray based on hit order using bitonic merge sort.
- Raycast against points (actually against triangulated mesh).
- Visualization of hit points.

### Future work:
- Generate viewsheds (panoramic screenshot of visible points).
- Load multiple LAZ files to cover bigger areas.
- Use database APIs to load model isntead of loading from disk.
- Select points to ignore based on classification.

## Usage 
To do a panoramic raycast:
1. Load a LAZ file from disk and wait for preprocessing to finish (might take around a minute depending on hardware).
2. Switch to `Render Mode: Visibility`. 
3. Press `Run Panorama` to do a panoramic raycast around the chosen origin.
4. Hit points are shown in white.

To repeat from another position, click `Clear Points` and set another origin using the input fields.  

## Controls

Movement: Drag middle mouse button to orbit the camera. Drag left mouse button to pan.

Settings:
- `Load file`: Load an LAZ file from the device.
- `Clear cache`: Clear any data for previously loaded files.
- `Samples X/Y`: Set the resolution of the raycast.
- `Origin X/Y/Z`: Set the origin of the raycast.
- `Start/End Theta/Phi`: Set spherical coordinates of the direction of the raycast.
- `Run Nodes`: Do a raycast against the nodes of the quadtree.
- `Sort Nodes`: After the node-raycast, sort nodes baded on hit order.
- `Run Points`: Do a raycast against the individual points.
- `Clear Points`: Clear raycast hit data.
- `Run Panorama`: Runs a raycast 360° around the origin. Each degree does one rayacast with the sample rate that ways set above.
- `Render Mode`:
  - `Color`: Points color based on LAZ file data.
  - `Visibility`: White for points hit by the raycast, black for others.
  - `Node assignment`: Quadtree node assignment for each point.
  - `Classification`: Point classification based on LAZ file data.
- `Render points`: Show model points.
- `Render mesh`: Render delauney mesh.
- `Render rays`: Render the raycast rays.
- `Render nodes`: Render quadtree nodes.

## Ueseful links
Websites used to aquire LAZ files of Airborne Laser Scans:
- https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=laserdaten (GER)
- https://www.opengeodata.nrw.de/produkte/geobasis/hm/3dm_l_las/3dm_l_las/ (GER)
- https://mapy.geoportal.gov.pl/imap/Imgp_2.html (POL)
- https://data.geobasis-bb.de/geobasis/daten/als/laz/ (GER)
- https://bb-viewer.geobasis-bb.de/?projection=EPSG:25833&center=447360,5735877&zoom=6&bglayer=1&layers=38 (GER)
- https://bb-viewer.geobasis-bb.de/ (GER)
- https://www.geoportal.de/search.html?q=laserscan (GER)

(only what I found, for regions that were interesting to me.)