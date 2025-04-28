from fastapi import FastAPI, File, UploadFile
import shutil
import exifread
import requests

def run_aircraft_check(image_path):
    img_tensor = load_image(image_path)
    class_name = classify_image(img_tensor, class_labels)

    if alert_if_military_or_fake(class_name):
        return {"status": "blocked", "reason": class_name}

    lat, lon = get_lat_lon(image_path)
    if not lat or not lon:
        return {"status": "error", "reason": "No GPS metadata"}

    # Call API to get aircraft info
    flight_info = get_flight_info(lat, lon)  # ← I'll help write this
    return {
        "status": "success",
        "classification": class_name,
        "coordinates": (lat, lon),
        "flight_info": flight_info
    }