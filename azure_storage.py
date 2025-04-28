<<<<<<< HEAD
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import streamlit as st

#connection details
connection_string = st.secrets["connection_string"] 
container_name1 = "user-input"
correctedpredictions_container = "correctedpredictions"

def upload_to_blob(image_file, blob_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name1)
        #upload image to storage account
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(image_file, overwrite=True)
        return f"File uploaded to {blob_name}"
    except Exception as e:
        print(f"Error uploading file to Storage Account: {e}")
        return None
from azure.storage.blob import BlobServiceClient

#image retraining mechanism
def list_blob_files():
    try:
        #connect to the storage
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name1)
        #list blobs in the container (files)
        blobs = container_client.list_blobs()
        return [blob.name for blob in blobs]
    
    except Exception as e:
        print(f"Error listing blobs: {e}")
        return []

def upload_corrected_image_to_blob(image_file, corrected_blob_name, corrected_class):
    try:
        #connect to the correctedpredictions storage account
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(correctedpredictions_container)
        #upload image to storage account with corrected class as metadata
        blob_client = container_client.get_blob_client(corrected_blob_name)
        
        #upload the image as a file with corrected classification as metadata
        blob_client.upload_blob(image_file, overwrite=True)
        blob_client.set_blob_metadata({"corrected_class": corrected_class})
        return f"Corrected image uploaded to {corrected_blob_name} with classification: {corrected_class}"
    except Exception as e:
        print(f"Error uploading corrected image to Storage Account: {e}")
=======
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os
import streamlit as st

#connection details
connection_string = st.secrets["connection_string"] 
container_name1 = "user-input"
correctedpredictions_container = "correctedpredictions"

def upload_to_blob(image_file, blob_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name1)
        #upload image to storage account
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(image_file, overwrite=True)
        return f"File uploaded to {blob_name}"
    except Exception as e:
        print(f"Error uploading file to Storage Account: {e}")
        return None
from azure.storage.blob import BlobServiceClient

#image retraining mechanism
def list_blob_files():
    try:
        #connect to the storage
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name1)
        #list blobs in the container (files)
        blobs = container_client.list_blobs()
        return [blob.name for blob in blobs]
    
    except Exception as e:
        print(f"Error listing blobs: {e}")
        return []

def upload_corrected_image_to_blob(image_file, corrected_blob_name, corrected_class):
    try:
        #connect to the correctedpredictions storage account
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(correctedpredictions_container)
        #upload image to storage account with corrected class as metadata
        blob_client = container_client.get_blob_client(corrected_blob_name)
        
        #upload the image as a file with corrected classification as metadata
        blob_client.upload_blob(image_file, overwrite=True)
        blob_client.set_blob_metadata({"corrected_class": corrected_class})
        return f"Corrected image uploaded to {corrected_blob_name} with classification: {corrected_class}"
    except Exception as e:
        print(f"Error uploading corrected image to Storage Account: {e}")
>>>>>>> 2476895c71763313186bb0c18917f7a5a6ff1428
        return None