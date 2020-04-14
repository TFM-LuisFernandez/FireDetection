import numpy as np
import cv2


class FireAnalizer:
    """Interfaz para obtener características de las regiones candidatos."""

    def __init__(self, number_objects: int, ratio_large: int):
        """
        Crea un discriminador de las regiones erroneamente segmentadas.

        :param number_objects: número de objetos segmentados
        :param ratio_large: proporción en la que hay que agrandar los contornos para la visualización
        """
        self.number_objects = number_objects
        self.ratio_large = ratio_large

    def features_fire(self, markers: np.array, rescale_image: np.array) -> dict:
        """
        Interfaz para recoger las características de las regiones de interés.

        :param markers: marcadores de las regiones segmentadas (objetos)
        :param rescale_image: imagen original reescalada (3 Canales)
        :returns: diccionario con las características del fuego que son relevantes para su clasificación
        """
        fire_features = {'contours': list(), 'bd': list(), 'area': list(), 'centroids': list(), 'mean': list()}

        for object in range(1, self.number_objects):
            mask_one_marker = cv2.convertScaleAbs(np.uint8(markers == object))
            _, thresh_one_marker = cv2.threshold(mask_one_marker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # cv2.imshow('object', thresh_one_marker)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # Media del color
            mean = cv2.mean(rescale_image, mask=thresh_one_marker)
            fire_features['mean'].append(sum(mean) / len(mean))

            # contornos
            contour, _ = cv2.findContours(thresh_one_marker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contour) > 0:
                # Reescalar contornos -> Visualización
                contour[0][:, :, 0] = contour[0][:, :, 0] * self.ratio_large
                contour[0][:, :, 1] = contour[0][:, :, 1] * self.ratio_large
                fire_features['contours'].append(np.array(contour))

                hull = []
                for i in range(len(contour)):
                    hull.append(cv2.convexHull(contour[i], False))

                # Desorden de los límites (boundary disorder)
                perimeter_cnt = cv2.arcLength(contour[0], True)
                perimeter_hull = cv2.arcLength(hull[0], True)
                if perimeter_cnt != 0:
                    fire_features['bd'].append(round(perimeter_hull / perimeter_cnt, 2))
                else:
                    fire_features['bd'].append(0.0)

                # Numero de pixeles (Area)
                fire_features['area'].append(len(cv2.findNonZero(thresh_one_marker)))

                # Centroides (Cx, Cy)
                moments = cv2.moments(contour[0])
                if moments['m00'] != 0:
                    (cx, cy) = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                else:
                    (cx, cy) = (0, 0)

                fire_features['centroids'].append((cx, cy))

        return fire_features
