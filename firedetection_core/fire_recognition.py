from firedetection_core.fire_preprocessor import GrayPreprocessing, RainbowPreprocessing
from firedetection_core.fire_segmentation import FireSegmentation
from firedetection_core.fire_analysis import FireAnalizer
from firedetection_core.fire_classification import FireClassifier

import numpy as np

RECOGNIZER_TYPES = {1: {'recognizer': GrayPreprocessing, 'description': 'gray thermal image recognizer (streaming)'},
                    2: {'recognizer': RainbowPreprocessing, 'description': 'rainbpw thermal image recognizer (local)'}}


class FireRecognizer:
    """Interfaz común a seguir por los reconocedores de fuego."""

    def recognize_image(self, buffer: list) -> dict:
        """Reconoce las regiones con fuego presentes en la imagen recibida."""
        raise NotImplementedError()


class Recognizer(FireRecognizer):
    """Reconocedor de las regiones con fuego en imagenes de escala de gris (3 Canales)."""

    def __init__(self, recognizer_type: int):
        """Crea un reconocedor de fuego en imagenes de grises."""
        if recognizer_type not in RECOGNIZER_TYPES:
            recognizer_types = [
                '{} - {}'.format(recognizer_type_key, RECOGNIZER_TYPES[recognizer_type_key]['description']) for
                recognizer_type_key in RECOGNIZER_TYPES.keys()]
            raise ValueError('Recognizer type not available, try one of: {}'.format(recognizer_types))

        if recognizer_type == 1:
            self.color_max = np.array([50, 10, 255], np.uint8)
            self.color_min = np.array([0, 0, 110], np.uint8)
        elif recognizer_type == 2:
            self.color_max = np.array([200, 10, 255], np.uint8)
            self.color_min = np.array([0, 0, 200], np.uint8)

        self.fire_preproccesor = RECOGNIZER_TYPES[recognizer_type]['recognizer'](enhacement=1.5, ratio_small=0.25)
        self.fire_segmentation = FireSegmentation(noise_size=3, color_min=self.color_min, color_max=self.color_max)
        self.fire_analysis = FireAnalizer(number_objects=0,
                                          ratio_large=int(1 / self.fire_preproccesor.small_to_large_image_size_ratio))
        self.fire_classification = FireClassifier(shape_threshold_min=0.69, shape_threshold_max=0.96,
                                                  area_threshold=150, mean_threshold=130.0)
                                                    # shape -> 0.75; 0.87; ara -> 600 (ratio_small=0.5

    def recognize_image(self, buffer: list) -> dict:
        """
        Reconoce las regiones de la imagen recibida (3 canales).

        :param buffer: buffer de 3 imágenes originales (3 canales)
        :returns: diccionario formado por los contornos del fuego y la imagen original reescalada
        """
        fire_detection = {'roi': None, 'rescale': None, 'features': None, 'detection': None}
        # buffer de imagenes preprocesadas
        preprocessing_images = list()

        # Preprocesamiento de todas las imagenes del buffer
        for image in buffer:
            preprocessing_image, rescale_image, roi_image = self.fire_preproccesor.preprocessing_image(image=image)
            preprocessing_images.append(preprocessing_image)

        fire_detection['roi'] = roi_image
        fire_detection['rescale'] = rescale_image

        # import cv2
        # cv2.imshow('buffer_prepocesamiento', cv2.hconcat([preprocessing_images[0], preprocessing_images[1], preprocessing_images[2]]))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # Segmentación de regiones candidatas (imagen actual) -> Filtrado temporal y de color
        markers, number_objects = self.fire_segmentation.segmentation_fire(buffer=preprocessing_images)
        if (markers is not None) and (number_objects > 0):
            # Extracción de características de cada una de las regiones candidatas
            self.fire_analysis.number_objects = number_objects
            features = self.fire_analysis.features_fire(markers=markers, rescale_image=fire_detection['rescale'])

            # claisificación de las regiones candidatas por sus caracteristicas
            fire_detection['features'] = self.fire_classification.classifying_fire(features=features)
        else:
            fire_detection['features'] = {'contours': list(), 'bd': list(), 'area': list(), 'centroids': list(),
                                          'mean': list()}

        if len(fire_detection['features']['contours']) > 0:
            fire_detection['detection'] = 'Fire Detected'
        else:
            fire_detection['detection'] = 'Undetected Fire'

        ####################################################################
        # import cv2
        #
        # alto, ancho, c = buffer[2].shape
        #
        # image2 = np.zeros((alto + 400, ancho, c), np.uint8)
        # image_copy = buffer[2].copy()
        #
        # try:
        #     for object in range(len(features['contours'])):
        #         # pintar parametros -> imagen 2
        #         cv2.drawContours(image_copy, features['contours'][object], -1, (0, 255, 0), 2)
        #         cv2.putText(image_copy, str(object),  # str(indexBD_actual.index(cnt)),
        #                     (features['centroids'][object][0] - 5, features['centroids'][object][1] - 5),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        #         cv2.putText(image2, 'bd: ' + str(features['bd'][object]), (20, 20 + object * 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        #         cv2.putText(image2, 'area: ' + str(features['area'][object]), (20, 40 + object * 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        #         cv2.putText(image2, 'media: ' + str(features['mean'][object]), (20, 60 + object * 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        #         cv2.putText(image2, '-----------------', (20, 80 + object * 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        #
        #
        #     cv2.imshow('prueba', cv2.vconcat([image_copy, image2]))
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        # except:
        #     pass

        return fire_detection
