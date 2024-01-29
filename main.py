"""
Workshop para la tech week Liverpool 2024
Cloud Automation
"""
import functions_framework
import pathlib
from vision_api import Vision
from bucket import Gcs


# Importar clases de ayuda
vision = Vision()
gcs = Gcs()


@functions_framework.cloud_event
def vision_api(cloud_event):

    # Definir variables a partir del evento que disparó
    # la Cloud Function
    data = cloud_event.data
    bucket = data["bucket"]
    fullname = data['name']
    name = pathlib.PurePath(fullname).name

    # Restringir a imágenes que se encuentren dentro del folder "original"
    if not fullname.startswith("original/"):
        print(f"{fullname} file is inside a folder that's not suitable for analysis")
    # Restringir a imagenes jepg
    if data["contentType"] != f"image/jpeg":
        raise RuntimeError(f"{name} file is not a jpeg image")
    # Descargar imagen que disparó la ejecución del Cloud Function
    gcs.download_to_filename(bucket, fullname, name)
    print(f'Image successfully downloaded from {bucket} bucket')
    # Interactuar con Cloud Vision API
    objects = vision.detect_objects(f"/tmp/{name}")
    print('Data successfully obteined from Cloud Vision API')
    # Procesar información obtenida de Cloud Vision API
    object, score, vertices = vision.process_data(objects)
    print('Data successfully processed')
    # Definir la metadata de la nueva imagen a partir de la
    # información procesada
    metadata = {
        'objeto': object,
        'score': score
    }
    # Editar imagen
    cropped_image = vision.crop_object(f"/tmp/{name}", vertices)
    print('Image successfully processed')
    # Respaldar imagen procesada en un bucket
    gcs.upload_to_bucket(bucket, name, cropped_image, metadata)
    print(f'Image successfully uploaded to {bucket} bucket')
