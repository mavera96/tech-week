"""
Workshop para la tech week Liverpool 2024
Cloud Automation
"""

import cv2
from google.cloud import vision

class Vision:
    """
    Clase para operar todo lo relacionado a la comunicación
    con Google Vision API
    """
    @staticmethod
    def detect_objects(image):
        """
        Función que envía la imagen a Google Vision API
        y recibe los datos encontrados de la misma.
        """

        # Abrir imagen
        with open(image, "rb") as image_file:
            content = image_file.read()
        # Inicializar cliente
        client = vision.ImageAnnotatorClient()
        # Ejecutar "Object localization" (detecta objetos en la imagen)
        # y guardar los datos obtenidos de Google Vision API
        img = vision.Image(content=content)
        objects = client.object_localization(image=img).localized_object_annotations

        # Devolver información sobre los objetos encontrados
        return objects

    @staticmethod
    def process_data(objects):
        """
        Función que procesa los datos recibidos de Google Vision API
        y devuelve la información necesaria.
        """

        vertices = []

        # Iterar los objetos encontrados en la imagen.
        # Por conveniencia solo se trabajará con el primer
        # objeto encontrado.
        for object_ in objects:
            # Obtener los vertices para gráficar.
            # Los vertices están normalizados.
            for vertex in object_.bounding_poly.normalized_vertices:
                tmp = (vertex.x, vertex.y)
                vertices.append(tmp)

            # Se devuelven los vértices e información
            # adicional como el nombre del objeto que se 
            # encontró y la probabilidad de acertividad.
            return object_.name, object_.score * 100, vertices

        return False

    @staticmethod
    def crop_object(image, vertices):
        """
        Función que modifica la imagen original
        y dibuja sobre ella un marco que encierra el objeto detectado.
        """

        original_image = cv2.imread(image)
        cropped_image = "cropped.jpg"

        # Obtener líneas y columnas de la imagen
        (number_of_rows, number_of_columns) = original_image.shape[:2]
        # Desnormalizar vertices
        v1_y = round(vertices[0][0] * number_of_columns)
        v1_x = round(vertices[0][1] * number_of_rows)
        v2_y  = round(vertices[2][0] * number_of_columns)
        v2_x = round(vertices[2][1] * number_of_rows)
        # Agregar un rectángulo en la imagen para
        # encuadrar el objeto encontrado en la imagen.
        cropped_img = cv2.rectangle(
            original_image,
            (v1_y, v1_x),
            (v2_y, v2_x),
            (0, 255, 0),
            8
        )
        # Guardar imagen
        cv2.imwrite(f"/tmp/{cropped_image}", cropped_img)

        # Rgresa el nombre de la imagen generada
        return cropped_image
