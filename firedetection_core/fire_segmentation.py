import cv2
import numpy as np


class FireSegmentation:
    """Detector de las regiones donde hay fuego en una imagen."""

    def __init__(self, noise_size: int, color_min: np.array, color_max: np.array):
        """
        Crea un reconocedor de fuego en una imagen.

        :param noise_size: tamaño del ruido esperando en la imagen
        :param color_min: límite inferiror para el filtro de color
        :param color_max: límite superiror para el filtro de color
        """
        self.color_min = color_min
        self.color_max = color_max
        self.structuring_element = np.ones((noise_size, noise_size), np.uint8)

    def segmentation_fire(self, buffer: list) -> tuple:
        """
        Localiza las regiones candidatas a ser fuego en la imagen.

        :param buffer: lista de imagenes preprocesadas HSV (3 Canales)
        :returns: tupla de los marcadores de los diferentes objetos y el número de objetos detectados
        """
        binary_images = list()

        # filtro de color HSV
        for image in buffer:
            binary_images.append(cv2.inRange(image, self.color_min, self.color_max))

        # cv2.imshow('filtro de color', cv2.hconcat([binary_images[0], binary_images[1], binary_images[2]]))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        if not np.all(binary_images[2] == 0):
            # Encontrar regiones que se hayan desplazado entre frames
            diff_2_1 = cv2.absdiff(binary_images[2], binary_images[1])
            diff_2_0 = cv2.absdiff(binary_images[2], binary_images[0])
            diff_image = cv2.absdiff(diff_2_1, diff_2_0)

            # cv2.imshow('filtro de color', diff_image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # Semillas de los objetos de primer plano -> Regiones que hayan pasado el filtro de color y
            # tengan desplazamiento entre imagenes
            seed_objects_image = np.zeros(diff_image.shape, np.uint8)
            contours, _ = cv2.findContours(binary_images[2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                region_image = np.zeros(diff_image.shape, np.uint8)
                cv2.drawContours(region_image, [cnt], -1, 255, -1)

                movement_region_image = cv2.bitwise_and(diff_image, diff_image, mask=region_image)
                # si la region tiene píxeles desplazados -> Region semilla
                if not np.all(movement_region_image == 0):
                    seed_objects_image = cv2.add(seed_objects_image, region_image)

            # Semilla para el fondo -> pixeles que no han superado el filtro de color en el frame actual
            fondo = cv2.bitwise_not(binary_images[2])
            fondo = cv2.erode(fondo, self.structuring_element, iterations=10)

            # cv2.imshow('filtro de color', cv2.hconcat([seed_objects_image, fondo]))
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # Definimos etiquetas
            number_objects, labels = cv2.connectedComponents(seed_objects_image)

            # Fondo tendrá la última etiqueta
            labels[fondo == 255] = number_objects + 1

            # marcadores
            markers = cv2.watershed(buffer[2], labels)

            # print('number_objects', number_objects)

            # Generate random colors
            # import random as rng
            # colors = []
            #
            # contours, _ = cv2.findContours(seed_objects_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #
            # for contour in contours:
            #     colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
            # # Create the result image
            # dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
            # # Fill labeled objects with random colors
            # for i in range(markers.shape[0]):
            #     for j in range(markers.shape[1]):
            #         index = markers[i, j]
            #         if index > 0 and index <= len(contours):
            #             dst[i, j, :] = colors[index - 1]
            # # Visualize the final image
            # cv2.imshow('Final Result', dst)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
        else:
            number_objects = 0
            markers = None


        return markers, number_objects
