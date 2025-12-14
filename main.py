import ee
import requests
import os
import numpy as np
from roboflow import Roboflow
from ultralytics import YOLO

rf = Roboflow(api_key="rQgVwkWjttExhgO7GqUO")
project = rf.workspace("solar-h55md").project("solar-panel-qcdgb")
version = project.version(4)
dataset = version.download("yolov8")

# Load a model
model = YOLO("yolov8n-seg.pt")

# Train the model
train_results = model.train(
    data="/content/Solar-Panel-4/data.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()


def get_satellite_image_as_png(lat, lon, start_date, end_date, output_filename='output_image.png', dimensions=512):
    """
    Fetches a cloud-filtered satellite image chip for a specific location
    and saves it as a PNG file.

    Args:
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center point.
        start_date (str): Start date for image search (YYYY-MM-DD).
        end_date (str): End date for image search (YYYY-MM-DD).
        output_filename (str): Name of the file to save the PNG to.
        dimensions (int): Size of the image (e.g., 512x512 pixels).
    """
    try:
        ee.Initialize()
        print("GEE initialized successfully.")
    except Exception as e:
        print(f"GEE initialization failed. Ensure 'earthengine authenticate' was run. Error: {e}")
        return

    point = ee.Geometry.Point(lon, lat)

    # 1. Filter the Sentinel-2 Image Collection
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(point) \
        .filterDate(start_date, end_date) \
        .sort('CLOUDY_PIXEL_PERCENTAGE') \
        .first() # Get the least cloudy image

    if not collection:
        print("No suitable images found for the given criteria.")
        return

    # 2. Define visualization parameters for RGB bands
    # These parameters scale the raw 16-bit data (0-10000) to 8-bit (0-255) for display
    vis_params = {
        'bands': ['B4', 'B3', 'B2'], # Red, Green, Blue bands
        'min': 0,
        'max': 3000, # A standard max value for good visual contrast
        'dimensions': [dimensions, dimensions],
        'format': 'png'
    }

    # 3. Get the image URL
    try:
        url = collection.getThumbURL(vis_params)
        print(f"Generated image download URL.")
    except ee.EEException as e:
        print(f"Error generating URL: {e}")
        return

    # 4. Use 'requests' to fetch the image and save to file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        print(f"\nSuccessfully saved image to {os.path.abspath(output_filename)}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

# import PIL
# --- Example Usage ---
if __name__ == '__main__':
    # Coordinates for a location (e.g., San Francisco)
    latitude = 37.7749
    longitude = -122.4194

    # Time window to search for images
    start = '2023-01-01'
    end = '2024-01-01'

    get_satellite_image_as_png(
        lat=latitude,
        lon=longitude,
        start_date=start,
        end_date=end,
        output_filename='san_francisco_chip.png',
        dimensions=1024
    )

results = model('san_francisco_chip.png')
results[0].show()


masks = results[0].masks

total_area_pixels = 0

for i, mask in enumerate(masks.data):
    mask_np = mask.cpu().numpy()
    area = np.sum(mask_np > 0)
    total_area_pixels += area

    print(f"Panel {i+1}: Area = {area} pixels²")

print(f"\nTotal panel area = {total_area_pixels} pixels²")

pixel_size_m = 0.05
area_m2 = total_area_pixels * (pixel_size_m ** 2)

print(f"Total area = {area_m2:.2f} m²")
estimated_kw = area_m2 / 6.5
print(f"Estimated solar capacity ≈ {estimated_kw:.2f} kW")
kw_installed = area_m2 / 6.5

print(f"Installed solar power ≈ {kw_installed:.2f} kW")
sun_hours = 5  # India average

daily_energy_kwh = kw_installed * sun_hours

print(f"Daily energy ≈ {daily_energy_kwh:.2f} kWh/day")
monthly_energy = daily_energy_kwh * 30
yearly_energy = daily_energy_kwh * 365

print(f"Monthly energy ≈ {monthly_energy:.2f} kWh/month")
print(f"Yearly energy ≈ {yearly_energy:.2f} kWh/year")
