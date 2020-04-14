import numpy as np
import cv2


class FireVisualize:
    """Interfaz para representar los resultados obtenidos en la detección de incendios"""

    def __init__(self):
        """Visualizador de los resultados de la detección de incendios en imágenes térmicas"""
        self.result_image = None
        self.mask_image = None

    def visualize_restult(self, image: np.array, contours: np.array, centroids: np.array) -> tuple:
        """
        Muestra el resultado de la detección en una ventana de OpenCV

        :param image: imagen original reescalada (3 Canales)
        :param contours: bounding boxes (contornos) de las regiones con fuego
        :param centroids: duplas con las coordenadas X, Y de los centroides de las regiones con fuego
        :returns: tupla de imagen original reescalada con contornos del fuego marcados en verde e imagen
        mascara con las regiones del fuego segmentadas
        """
        self.mask_image = np.zeros_like(image)
        self.result_image = image.copy()
        binary_image = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        if (contours is not None) and (len(contours) > 0):
            for cnt in contours:
                cv2.drawContours(self.result_image, cnt, -1, (0, 255, 0), 2)

                cv2.drawContours(binary_image, cnt, -1, 255, -1)
                self.mask_image = cv2.add(self.mask_image, cv2.bitwise_and(image, image, mask=binary_image))
                cv2.drawContours(self.mask_image, cnt, -1, (0, 255, 0), 2)
                for cntd in centroids:
                    cv2.circle(self.mask_image, cntd, 4, (255, 0, 0), -1)
                    cv2.circle(self.result_image, cntd, 4, (255, 0, 0), -1)

                textsize = cv2.getTextSize('FIRE DETECTED', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                cv2.putText(self.mask_image, 'FIRE DETECTED',
                            (int((image.shape[1] - textsize[0]) / 2), 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        else:
            textsize = cv2.getTextSize('UNDETECTED FIRE', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            cv2.putText(self.mask_image, 'UNDETECTED FIRE',
                        (int((image.shape[1] - textsize[0]) / 2),
                         int((image.shape[0] + textsize[1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # cv2.namedWindow('Fire Detection', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('Fire Detection', cv2.hconcat([self.result_image, self.mask_image]))
        # cv2.waitKey()

        return self.result_image, self.mask_image
