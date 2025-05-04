import piexif
from piexif.helper import UserComment
from PIL import Image
from datetime import datetime

# Load the image
img_path = "\Users\sccla\Documents\met\level 6\Final Year Project\FYP Coding Project\test images\commercial2.jpg"
img = Image.open(img_path)

# Define GPS coordinates (in EXIF rational format)
def to_deg(value, ref):
    deg = int(value)
    min_float = (value - deg) * 60
    min = int(min_float)
    sec = int((min_float - min) * 60 * 100)
    return ((deg, 1), (min, 1), (sec, 100)), ref

lat, lat_ref = to_deg(37.7749, "N")  # Example: San Francisco
lon, lon_ref = to_deg(122.4194, "W")

# Define DateTimeOriginal
now = datetime.now().strftime("%Y:%m:%d %H:%M:%S")

# Create EXIF data
exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = now.encode()
exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = lat_ref.encode()
exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = lat
exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = lon_ref.encode()
exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = lon

# Inject EXIF metadata
exif_bytes = piexif.dump(exif_dict)
img.save("injected_image.jpg", exif=exif_bytes)
