import joblib
import pandas as pd
import requests

from shapely.geometry import Polygon
from pyproj import Transformer

# تحميل الموديل المدرب
from pathlib import Path

model_path = Path(__file__).parent / "model.pkl"

model = joblib.load(model_path)
# ==========================================
# Calculate Farm Area
# ==========================================

def calculate_area(p1, p2, p3, p4):

    polygon = Polygon([p1, p2, p3, p4])

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32636",
        always_xy=True
    )

    projected_points = []

    for lon, lat in polygon.exterior.coords:
        x, y = transformer.transform(lon, lat)
        projected_points.append((x, y))

    projected_polygon = Polygon(projected_points)

    area_square_meter = projected_polygon.area

    area_hectare = area_square_meter / 10000

    return area_hectare


# ===== تجربة =====
p1 = (31.10, 30.10)
p2 = (31.20, 30.10)
p3 = (31.20, 30.20)
p4 = (31.10, 30.20)

print(calculate_area(p1, p2, p3, p4))
# ==========================================
# Get Weather Data
# ==========================================

def get_weather(lat, lon):

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}"
        f"&longitude={lon}"
        f"&current=temperature_2m"
        f"&daily=precipitation_sum"
        f"&timezone=auto"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Weather API Error")

    data = response.json()

    temperature = data["current"]["temperature_2m"]
    rainfall = data["daily"]["precipitation_sum"][0]

    return temperature, rainfall
temperature, rainfall = get_weather(30.0444, 31.2357)

print("Temperature:", temperature)
print("Rainfall:", rainfall)
# ==========================================
# Get Soil Data
# ==========================================

def get_soil_data(lat, lon):

    url = (
        f"https://rest.isric.org/soilgrids/v2.0/properties/query"
        f"?lat={lat}"
        f"&lon={lon}"
        f"&property=phh2o"
        f"&property=clay"
        f"&property=soc"
        f"&depth=0-5cm"
        f"&value=mean"
    )

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Soil API Error")

    data = response.json()

    layers = data["properties"]["layers"]

    soil = {}

    for layer in layers:
        name = layer["name"]
        value = layer["depths"][0]["values"]["mean"]

        if value is None:
            value = 0

        soil[name] = value

    return soil

# ===== تجربة بيانات التربة =====
soil = get_soil_data(30.0444, 31.2357)

print("\nSoil Data:")
print(soil)


def estimate_wheat_yield(p1, p2, p3, p4):

    area = calculate_area(p1, p2, p3, p4)

    lat = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
    lon = (p1[0] + p2[0] + p3[0] + p4[0]) / 4

    temperature, rainfall = get_weather(lat, lon)

    soil = get_soil_data(lat, lon)

    sample = pd.DataFrame({
        "Area": ["Egypt"],
        "Item": ["Wheat"],
        "Year": [2026],
        "average_rain_fall_mm_per_year": [rainfall],
        "avg_temp": [temperature]
    })

    prediction = model.predict(sample)[0]

    print("\n========== RESULT ==========")
    print("Farm Area (ha):", area)
    print("Temperature:", temperature)
    print("Rainfall:", rainfall)
    print("Soil:", soil)
    print("Predicted Wheat Yield:", prediction)

    return prediction


print("\n========== FINAL RESULT ==========")

estimate_wheat_yield(p1, p2, p3, p4)