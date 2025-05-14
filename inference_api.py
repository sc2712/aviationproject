import requests
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import streamlit as st
from datetime import datetime
import io
import toml

secrets = toml.load(".streamlit/secrets.toml")

#API Keys
AVIATIONSTACK_API_KEY = st.secrets["AVIATIONSTACK_API_KEY"] 
OPENSKY_USERNAME = st.secrets["OPENSKY_USERNAME"]
OPENSKY_PASSWORD = st.secrets["OPENSKY_PASSWORD"]

#load model and class labels
model = tf.keras.models.load_model("aircraft_classifier.keras")
with open("class_labels.txt", "r") as f:
    class_labels = [line.strip() for line in f]

#functions
#preprocess input image
def preprocess_image(image_path, img_size=(224, 224)):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

#classify image
def classify_image(img_tensor):
    prediction = model.predict(img_tensor)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    class_name = class_labels[class_index]
    print(f"Prediction: {class_name} | Confidence: {confidence:.2f}")
    return class_name, confidence

#check for fake or military aircraft
def handle_inference_decision(class_name):
    class_name_lower = class_name.lower()
    if "fake" in class_name_lower:
        return "ALERT: FAKE aircraft detected! Further analysis recommended."
    elif "military" in class_name_lower:
        return "WARNING: Military aircraft detected. Tracking suspended."
    return None

def check_alert(class_name):
    return class_name.startswith('fake') or 'military' in class_name.lower()

#extract metadata from input image
def extract_metadata(image_file):
    image_file.seek(0)  # Reset file pointer
    img = Image.open(io.BytesIO(image_file.read()))
    exif_data = img._getexif()
    image_file.seek(0)

    if not exif_data:
        print("No EXIF data found.")
        return None, None

    metadata = {}
    for tag, value in exif_data.items():
        decoded = TAGS.get(tag, tag)
        metadata[decoded] = value


    # Extract timestamp
    datetime_original = metadata.get("DateTimeOriginal")
    if datetime_original:
        try: #
            dt_obj = datetime.strptime(datetime_original, "%Y:%m:%d %H:%M:%S") #
            print("Parsed DateTimeOriginal:", dt_obj) #
        except ValueError as e: #
            print(f"Error parsing DateTimeOriginal: {e}") #
            dt_obj = None #
        #print("Found DateTimeOriginal:", datetime_original)
    else:
        print("DateTimeOriginal not found.")    
        dt_obj = None #

    # Extract GPS data
    gps_info = metadata.get("GPSInfo")
    if gps_info:
        gps_data = {}
        for key in gps_info.keys():
            decode = GPSTAGS.get(key, key)
            gps_data[decode] = gps_info[key]

        lat = _convert_to_degrees(gps_data.get("GPSLatitude"), gps_data.get("GPSLatitudeRef"))
        lon = _convert_to_degrees(gps_data.get("GPSLongitude"), gps_data.get("GPSLongitudeRef"))
    else:
        lat, lon = None, None

    return dt_obj, (lat, lon)

def _convert_to_degrees(value, ref):
    if value is None:
        return None
    degrees, minutes, seconds = value
    decimal = degrees[0]/degrees[1] + minutes[0]/minutes[1]/60 + seconds[0]/seconds[1]/3600
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal

def extract_gps_from_image(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data is None:
            print("No EXIF metadata found.")
            return None
        gps_info = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        if not gps_info:
            print("No GPS metadata found.")
            return None

        def convert_to_degrees(value):
            d, m, s = value
            return d[0]/d[1] + (m[0]/m[1])/60 + (s[0]/s[1])/3600
    
        lat = convert_to_degrees(gps_info['GPSLatitude'])
        if gps_info['GPSLatitudeRef'] != 'N':
            lat = -lat
        lon = convert_to_degrees(gps_info['GPSLongitude'])
        if gps_info['GPSLongitudeRef'] != 'E':
            lon = -lon
        return lat, lon

    except Exception as e:
        print(f"Error extracting GPS: {e}")
        return None

# import math

# def haversine_distance(lat1, lon1, lat2, lon2):
#     # Radius of the Earth in kilometers
#     R = 6371.0
#     # Convert coordinates from degrees to radians
#     phi1, phi2 = math.radians(lat1), math.radians(lat2)
#     delta_phi = math.radians(lat2 - lat1)
#     delta_lambda = math.radians(lon2 - lon1)
#     # Haversine formula
#     a = math.sin(delta_phi / 2) ** 2 + \
#         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     return R * c  # Distance in kilometers

# def lookup_flight_by_location(lat, lon, max_distance_km=100):
#     opensky_url = "https://opensky-network.org/api/states/all"
#     try:
#         response = requests.get(opensky_url, auth=(OPENSKY_USERNAME, OPENSKY_PASSWORD))
#         response.raise_for_status()
#         data = response.json()
#         min_distance = float("inf")
#         closest_flight = None
#         for state in data.get('states', []):
#             flight_lon = state[5]
#             flight_lat = state[6]
#             if flight_lon is None or flight_lat is None:
#                 continue
#             distance = haversine_distance(lat, lon, flight_lat, flight_lon)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_flight = {
#                     "callsign": state[1].strip() if state[1] else "N/A",
#                     "origin_country": state[2],
#                     "longitude": flight_lon,
#                     "latitude": flight_lat,
#                     "velocity": state[9],
#                     "altitude": state[7],
#                     "distance_km": distance
#                 }
#         if closest_flight and closest_flight["distance_km"] <= max_distance_km:
#             return closest_flight
#     except requests.RequestException as e:
#         print(f"OpenSky Request Error: {e}")
#     except ValueError as e:
#         print(f"JSON Decoding Error: {e}")
#     return None


def lookup_flight_by_location(lat, lon):
    opensky_url = "https://opensky-network.org/api/states/all"
    try:
        response = requests.get(opensky_url)
        data = response.json()
        min_distance = float("inf")
        closest_flight = None
        for state in data['states']:
            flight_lon = state[5]
            flight_lat = state[6]
            if flight_lon is None or flight_lat is None:
                continue
            distance = ((lat - flight_lat) ** 2 + (lon - flight_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_flight = {
                    "callsign": state[1].strip(),
                    "origin_country": state[2],
                    "longitude": flight_lon,
                    "latitude": flight_lat,
                    "velocity": state[9],
                    "altitude": state[7]
                }
        if closest_flight:
            return closest_flight
    except Exception as e:
        print(f"OpenSky Error: {e}")
    return None


# def lookup_flight_by_metadata(lat, lon, dt_obj):
#     try:
def lookup_flight_by_metadata(dt: datetime, lat, lon, distance_km=100):
    try:
        # Convert datetime to UNIX timestamp
        timestamp = int(dt.timestamp())

        # Step 1: Get ICAO24 using /tracks/all
        track_url = "https://opensky-network.org/api/tracks/all"
        track_params = {
            #"time": int(dt_obj.timestamp()),
            "lat": lat,
            "lon": lon
        }

        track_resp = requests.get(track_url, params=track_params, auth=(OPENSKY_USERNAME, OPENSKY_PASSWORD))
        track_resp.raise_for_status()
        track_data = track_resp.json()

        icao24 = track_data.get("icao24")
        if not icao24:
            print("No ICAO24 found in track data.")
            return None

        print(f"Found ICAO24: {icao24}")

        # if isinstance(dt_obj, float):  # Already a timestamp
        #     timestamp = dt_obj
        # else:
        #     timestamp = dt_obj.timestamp()

        # begin = int(timestamp) - 900  # 15 mins before
        # end = int(timestamp) + 900    # 15 mins after
        begin = timestamp - 900  # 15 mins before
        end = timestamp + 900    # 15 mins after

        flight_url = "https://opensky-network.org/api/flights/aircraft"
        flight_params = {
            "icao24": icao24,
            "begin": begin,
            "end": end
        }

        flight_resp = requests.get(flight_url, params=flight_params, auth=(OPENSKY_USERNAME, OPENSKY_PASSWORD))
        flight_resp.raise_for_status()
        flights = flight_resp.json()

        if flights:
            return {
                "icao24": icao24,
                "callsign": flights[0].get("callsign"),
                "est_departure_airport": flights[0].get("estDepartureAirport"),
                "est_arrival_airport": flights[0].get("estArrivalAirport"),
                "departure_time": flights[0].get("firstSeen"),
                "arrival_time": flights[0].get("lastSeen")
            }

        print("No flights found for ICAO24 in interval.")
        return None

    except Exception as e:
        print(f"Error getting flight details: {e}")
        return None





# def lookup_flight_by_metadata(datetime_original, lat, lon):
#     if not (datetime_original and lat and lon):
#         print("Missing metadata for flight lookup.")
#         return None

#     try:
#         dt_obj = datetime.strptime(datetime_original, "%Y:%m:%d %H:%M:%S")
#     except ValueError as ve:
#         print(f"Datetime parsing error: {ve}")
#         return None

#     url = (
#         f"http://api.aviationstack.com/v1/flights?"
#         f"access_key={AVIATIONSTACK_API_KEY}"
#         f"&limit=100&flight_date={dt_obj.date()}"
#     )

#     try:
#         response = requests.get(url)
#         data = response.json()

#         if "data" in data:
#             closest_flight = None
#             min_distance = float("inf")
#             for flight in data["data"]:
#                 dep = flight.get("departure", {})
#                 dep_lat, dep_lon = dep.get("latitude"), dep.get("longitude")
#                 if dep_lat is None or dep_lon is None:
#                     continue

#                 distance = ((lat - dep_lat) ** 2 + (lon - dep_lon) ** 2) ** 0.5
#                 if distance < min_distance:
#                     min_distance = distance
#                     closest_flight = flight

#             if closest_flight:
#                 return {
#                     "airline": closest_flight['airline']['name'],
#                     "flight_number": closest_flight['flight']['iata'],
#                     "departure_airport": closest_flight['departure']['airport'],
#                     "arrival_airport": closest_flight['arrival']['airport'],
#                     "status": closest_flight['flight_status'],
#                     "timestamp": closest_flight['flight_date']
#                 }
#     except Exception as e:
#         print(f"Flight metadata lookup error: {e}")
#     return None