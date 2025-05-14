import streamlit as st
from model_logic import keras_load_model, preprocess_image, classify_image
from inference_api import handle_inference_decision, extract_gps_from_image, lookup_flight_by_location, extract_metadata, lookup_flight_by_metadata
from azure_storage import upload_to_blob, upload_corrected_image_to_blob
from datetime import datetime
from azure.storage.blob import BlobServiceClient

#connection details for the "correctedpredictions" storage account
connection_string = st.secrets["connection_string"] 
correctedpredictions_container = "correctedpredictions"

#app layout
st.set_page_config(page_title="Aircraft Classifier", layout="centered")
st.title("Aircraft Image Classifier")
st.write("Upload an image to identify aircraft and search for related flights.")

uploaded_file = st.file_uploader("Upload an aircraft image", type=["jpg", "jpeg", "png"])

MAX_IMAGE_SIZE_MB = 5  
#check if the image size is below the limit
def check_image_size(image_file):
    image_size = image_file.size / (1024 * 1024)  #convert bytes to MB
    return image_size <= MAX_IMAGE_SIZE_MB
if uploaded_file is not None:
    metadata = extract_metadata(uploaded_file)

#upload image to azure storage
    blob_name = f"user_{uploaded_file.name}"
    upload_response = upload_to_blob(uploaded_file, blob_name)
    if upload_response:
        st.success(f"Image uploaded to Azure Storage: {upload_response}")
    else:
       st.error("Failed to upload image to Storage.")

#upload file into classifier
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image")
        st.write("\nClassifying...")
        model = keras_load_model("aircraft_classifier.keras")
        img_tensor = preprocess_image(uploaded_file)
        class_name, confidence = classify_image(model, img_tensor)
        st.success(f"Prediction: {class_name} ({confidence * 100:.2f}% confidence)")

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
        
    col1, col2 = st.columns(2)

        #GPS section
    st.subheader("Flight Lookup Based on Location")
        #save uploaded file temporarily
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
           
        
        with col1:
            if st.button("Find Nearby Flights from Location"):
                if lat and lon:
                    closest_flight = lookup_flight_by_location(lat, lon)
                    if closest_flight:
                        st.success(f"Closest flight: {closest_flight['callsign']} from {closest_flight['origin_country']}")
                        st.write(f"Location: ({closest_flight['latitude']:.4f}, {closest_flight['longitude']:.4f})")
                    else:
                        st.error("No flights found nearby.")

            
            #extract EXIF metadata
                st.subheader("Flight Lookup Based on Date / Time")
                dt, coords = extract_metadata(uploaded_file)
                st.write(f"Timestamp: {dt}" if dt else "No timestamp found.")
                st.write(f"Location: {coords}" if coords else "No GPS data found.")   
                
                #convert timestamp to datetime
                if dt:
                    dt_obj = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
                    alert_message = handle_inference_decision(class_name)
                    if alert_message:
                        st.error(alert_message)
                        #if it's military, stop execution of the GPS lookup
                        if class_name == "Military":
                            st.info("Military aircraft detected. GPS lookup will be skipped.")
                            st.stop()
                    else:
                        st.info("Aircraft appears normal. Proceed with tracking or info lookup.")

                #if metatdata found, look up nearest flights
                    if st.button("Find Nearby Flights using Date and Time"):
                        if lat and lon:
                            closest_flight = lookup_flight_by_location(lat, lon)
                            if closest_flight:
                                st.success(f"Closest flight: {closest_flight['callsign']} from {closest_flight['origin_country']}")
                                st.write(f"Location: ({closest_flight['latitude']:.4f}, {closest_flight['longitude']:.4f})")
                            else:
                                st.error("No flights found nearby.")

            with col2:        
                st.warning("If no Datetime metadata is found, please enter manually:")
                time_input = st.time_input("Select the time the image was taken (UTC)")
                date_input = st.date_input("Select the date the image was taken")

                if st.button("Find Nearby Flights"):
                    if time_input and date_input and lat and lon:
                        # manual_dt_str = f"{date_input.strftime('%Y:%m:%d')} {time_input.strftime('%H:%M:%S')}"
                        # closest_flight = lookup_flight_by_metadata(manual_dt_str, lat, lon)
                        manual_dt = datetime.combine(date_input, time_input)
                        closest_flight = lookup_flight_by_metadata(manual_dt, lat, lon)
                        if closest_flight:
                            st.success(f"Closest flight on {closest_flight['timestamp']}:")
                            st.write(f"{closest_flight['airline']} flight {closest_flight['flight_number']}")
                            st.write(f"From {closest_flight['departure_airport']} to {closest_flight['arrival_airport']}")
                            st.write(f"Status: {closest_flight['status']}")
                        else:
                            st.error("No flight match found for the given date/time and location.")
                    else:
                        st.error("Please enter full date, time, and valid coordinates.") 
            