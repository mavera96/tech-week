"""
Workshop para la tech week Liverpool 2024
Cloud Automation
"""

from google.cloud import storage

class Gcs:
    """
    Clase para operar lo relacionado a la comunicación
    con los buckets de gcp
    """
    @staticmethod
    def download_to_filename(bucket_name, bkt_filename, lcl_filename):
       """
       Función para descargar, desde el bucket de GCP, imagen proporcionada por
       el usuario
       """
       # Iniciar cliente de GCS
       storage_client = storage.Client()
       # Definir bucket del cual se descargará la imagen
       bucket = storage_client.bucket(bucket_name)
       # Definir el blob
       blob = bucket.blob(bkt_filename)
       # Descargar imagen
       blob.download_to_filename(f"/tmp/{lcl_filename}")

    @staticmethod
    def upload_to_bucket(bucket_name, bkt_filename, lcl_filename, metadata):
       """
       Función para suber, a bucket de GCP, imagen procesada
       con metadata adicional
       """
       # Iniciar cliente de GCS
       storage_client = storage.Client()
       # Definir bucket al cual se subirá la imagen
       bucket = storage_client.bucket(bucket_name)
       # Definir el blob y su metadata en base a los resultados
       # dobtenidos de Cloud Vision Api
       blob = bucket.blob(f"cropped/{bkt_filename}")
       blob.metadata = metadata
       # Subir imagen
       blob.upload_from_filename(f"/tmp/{lcl_filename}")
