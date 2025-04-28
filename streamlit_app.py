<<<<<<< HEAD
import streamlit as st
from model_logic import load_model, preprocess_image, classify_image
from inference_api import handle_inference_decision, extract_gps_from_image, lookup_flight_by_location
from azure_storage import upload_to_blob, upload_corrected_image_to_blob
from azure.storage.blob import BlobServiceClient
import io

#connection details for the "correctedpredictions" storage account
connection_string = st.secrets["connection_string"] 
correctedpredictions_container = "correctedpredictions"

#app layout
st.set_page_config(page_title="Aircraft Classifier", layout="centered")
st.title("Aircraft Image Classifier")

uploaded_file = st.file_uploader("Upload an aircraft image", type=["jpg", "jpeg", "png"])

MAX_IMAGE_SIZE_MB = 5  
#check if the image size is below the limit
def check_image_size(image_file):
    image_size = image_file.size / (1024 * 1024)  #convert bytes to MB
    return image_size <= MAX_IMAGE_SIZE_MB
if uploaded_file is not None:

#upload image to azure storage
    blob_name = f"user_{uploaded_file.name}"
    upload_response = upload_to_blob(uploaded_file, blob_name)
    if upload_response:
        st.success(f"Image uploaded to Azure Storage: {upload_response}")
    else:
       st.error("Failed to upload image to Storage.")

#upload file into classifier
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("\nClassifying...")
        model = load_model("aircraft_classifier.keras")
        img_tensor = preprocess_image(uploaded_file)
        class_name, confidence = classify_image(model, img_tensor)
        st.success(f"Prediction: {class_name} ({confidence * 100:.2f}% confidence)")

        alert_message = handle_inference_decision(class_name)
        if alert_message:
            st.error(alert_message)
            # If it's military, stop execution of the GPS lookup
            if class_name == "Military":
                st.info("Military aircraft detected. GPS lookup will be skipped.")
                st.stop()
        else:
            st.info("Aircraft appears normal. Proceed with tracking or info lookup.")

        #GPS section
        st.subheader("Flight Lookup Based on Location")
        #Save uploaded file temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

    #extract gps data
        gps_coords = extract_gps_from_image("temp_image.jpg")
        if gps_coords:
            lat, lon = gps_coords

    #print status
            st.success(f"GPS Data found: Latitude {lat:.6f}, Longitude {lon:.6f}")
        else:
            st.warning("No GPS metadata found. Please enter manually:")
            lat = st.number_input("Latitude:", format="%.6f")
            lon = st.number_input("Longitude:", format="%.6f")

    #if metatdata found, look up nearest flights
        if st.button("Find Nearby Flights"):
            if lat and lon:
                closest_flight = lookup_flight_by_location(lat, lon)
                if closest_flight:
                    st.success(f"Closest flight: {closest_flight['callsign']} from {closest_flight['origin_country']}")
                    st.write(f"Location: ({closest_flight['latitude']:.4f}, {closest_flight['longitude']:.4f})")
                else:
                    st.error("No flights found nearby.")
            else:
                st.error("Please provide valid coordinates.")

        #user correction for improving model prediction
        st.subheader("Correct the Classification (Optional)")

        #set to prediction
        correct_classification = st.radio(
            "Was the prediction correct?",
            ("Real Private Aircraft", "Real Commercial Aircraft", "Military Aircraft", "Fake Aircraft"),
            index=0 if class_name == "Real" else 1,
        )
        #button to submit correction
        if st.button("Submit Correction"):
            if correct_classification != class_name:
                
                #if the classification is corrected, upload the image with the corrected prediction to storage account
                corrected_image_name = f"corrected_{uploaded_file.name}"
                upload_response = upload_corrected_image_to_blob(uploaded_file, corrected_image_name, correct_classification)
                
                #feedback for the user
                if upload_response:
                    st.success(f"Correction submitted: {correct_classification}")
                    st.success(f"Corrected image uploaded to Azure Storage: {corrected_image_name}")
                else:
                    st.error("Failed to upload corrected image.")
            else:
                st.info("No correction needed. The prediction is correct.")
    else:
        st.info("Please upload an image to start.")
=======
import streamlit as st
from model_logic import load_model, preprocess_image, classify_image
from inference_api import handle_inference_decision, extract_gps_from_image, lookup_flight_by_location
from azure_storage import upload_to_blob, upload_corrected_image_to_blob
from azure.storage.blob import BlobServiceClient
import io

#connection details for the "correctedpredictions" storage account
connection_string = st.secrets["connection_string"] 
correctedpredictions_container = "correctedpredictions"

#app layout
st.set_page_config(page_title="Aircraft Classifier", layout="centered")
st.title("Aircraft Image Classifier")

uploaded_file = st.file_uploader("Upload an aircraft image", type=["jpg", "jpeg", "png"])

MAX_IMAGE_SIZE_MB = 5  
#check if the image size is below the limit
def check_image_size(image_file):
    image_size = image_file.size / (1024 * 1024)  #convert bytes to MB
    return image_size <= MAX_IMAGE_SIZE_MB
if uploaded_file is not None:

#upload image to azure storage
    blob_name = f"user_{uploaded_file.name}"
    upload_response = upload_to_blob(uploaded_file, blob_name)
    if upload_response:
        st.success(f"Image uploaded to Azure Storage: {upload_response}")
    else:
       st.error("Failed to upload image to Storage.")

#upload file into classifier
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("\nClassifying...")
        model = load_model("aircraft_classifier.keras")
        img_tensor = preprocess_image(uploaded_file)
        class_name, confidence = classify_image(model, img_tensor)
        st.success(f"Prediction: {class_name} ({confidence * 100:.2f}% confidence)")

        alert_message = handle_inference_decision(class_name)
        if alert_message:
            st.error(alert_message)
            # If it's military, stop execution of the GPS lookup
            if class_name == "Military":
                st.info("Military aircraft detected. GPS lookup will be skipped.")
                st.stop()
        else:
            st.info("Aircraft appears normal. Proceed with tracking or info lookup.")

        #GPS section
        st.subheader("Flight Lookup Based on Location")
        #Save uploaded file temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

    #extract gps data
        gps_coords = extract_gps_from_image("temp_image.jpg")
        if gps_coords:
            lat, lon = gps_coords

    #print status
            st.success(f"GPS Data found: Latitude {lat:.6f}, Longitude {lon:.6f}")
        else:
            st.warning("No GPS metadata found. Please enter manually:")
            lat = st.number_input("Latitude:", format="%.6f")
            lon = st.number_input("Longitude:", format="%.6f")

    #if metatdata found, look up nearest flights
        if st.button("Find Nearby Flights"):
            if lat and lon:
                closest_flight = lookup_flight_by_location(lat, lon)
                if closest_flight:
                    st.success(f"Closest flight: {closest_flight['callsign']} from {closest_flight['origin_country']}")
                    st.write(f"Location: ({closest_flight['latitude']:.4f}, {closest_flight['longitude']:.4f})")
                else:
                    st.error("No flights found nearby.")
            else:
                st.error("Please provide valid coordinates.")

        #user correction for improving model prediction
        st.subheader("Correct the Classification (Optional)")

        #set to prediction
        correct_classification = st.radio(
            "Was the prediction correct?",
            ("Real Private Aircraft", "Real Commercial Aircraft", "Military Aircraft", "Fake Aircraft"),
            index=0 if class_name == "Real" else 1,
        )
        #button to submit correction
        if st.button("Submit Correction"):
            if correct_classification != class_name:
                
                #if the classification is corrected, upload the image with the corrected prediction to storage account
                corrected_image_name = f"corrected_{uploaded_file.name}"
                upload_response = upload_corrected_image_to_blob(uploaded_file, corrected_image_name, correct_classification)
                
                #feedback for the user
                if upload_response:
                    st.success(f"Correction submitted: {correct_classification}")
                    st.success(f"Corrected image uploaded to Azure Storage: {corrected_image_name}")
                else:
                    st.error("Failed to upload corrected image.")
            else:
                st.info("No correction needed. The prediction is correct.")
    else:
        st.info("Please upload an image to start.")
>>>>>>> 2476895c71763313186bb0c18917f7a5a6ff1428
