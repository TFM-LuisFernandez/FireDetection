from PIL import Image, ImageEnhance
import numpy as np
import cv2


class PreprocessingImage:
    """Interfaz para el preprocesamiento de la imágen térmica."""

    def preprocessing_image(self, image: np.array) -> tuple:
        """Mejora las carácterísticas de la imagen que ayuden en la detección del foco del incendio."""
        raise NotImplementedError()


class GrayPreprocessing(PreprocessingImage):
    """Mejora de las características de las regiones con fuego en imagenes de escala de gris."""

    def __init__(self, enhacement: float, ratio_small: float):
        """Preprocesamiento de la imágen térmica."""
        self.enhacement = enhacement
        self.small_to_large_image_size_ratio = ratio_small
        self.grid_size = (5, 5)

    def preprocessing_image(self, image: np.array) -> tuple:
        """
        Mejora las carácterísticas de la imagen que ayuden en la detección del foco del incendio.

        :param image: imagen original (3 Canalaes)
        :returns: tupla de imagen preprocesada con las regiones de interés destacadas y la imagen original reescalada
        """
        reescale_image = cv2.resize(image,  # original image
                                    (0, 0),  # set fx and fy, not the final size
                                    fx=self.small_to_large_image_size_ratio,
                                    fy=self.small_to_large_image_size_ratio,
                                    interpolation=cv2.INTER_LINEAR)

        hsv_image = cv2.cvtColor(reescale_image, cv2.COLOR_BGR2HSV)

        blur_image = cv2.GaussianBlur(hsv_image, self.grid_size, 0)

        h, s, v = cv2.split(blur_image)

        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=self.grid_size)
        clahe_s_image = clahe.apply(s)

        # Enhacement contrast image
        enhacer = ImageEnhance.Contrast(Image.fromarray(clahe_s_image))
        enhacement_contrast_s_image = np.array(enhacer.enhance(self.enhacement))

        prep_image = cv2.merge([h, enhacement_contrast_s_image, v])

        return prep_image, reescale_image, image


class RainbowPreprocessing(PreprocessingImage):
    """Mejora de las características de las regiones con fuego en imagenes Rainbow."""

    def __init__(self, enhacement: float, ratio_small: float):
        """Preprocesamiento de la imágen térmica."""
        self.enhacement = enhacement
        self.small_to_large_image_size_ratio = ratio_small
        self.grid_size = (5, 5)
        self.reescale_image = None
        self.prep_image = None

    def preprocessing_image(self, image: np.array) -> tuple:
        """
        Mejora las carácterísticas de la imagen que ayuden en la detección del foco del incendio.

        :param image: imagen original (3 Canalaes)
        :returns: tupla de imagen preprocesada con las regiones de interés destacadas y la imagen original reescalada
        """
        roi_color = image[0:512 - 22, 0:640 - 45]
        # roi_ocr = image[5:25, 640 - 48:640 - 2]

        reescale_image = cv2.resize(roi_color,  # original image
                                    (0, 0),  # set fx and fy, not the final size
                                    fx=self.small_to_large_image_size_ratio,
                                    fy=self.small_to_large_image_size_ratio,
                                    interpolation=cv2.INTER_LINEAR)

        hsv_image = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

        res_image_hsv = cv2.resize(hsv_image,  # original image
                                   (0, 0),  # set fx and fy, not the final size
                                   fx=self.small_to_large_image_size_ratio,
                                   fy=self.small_to_large_image_size_ratio,
                                   interpolation=cv2.INTER_LINEAR)

        blur_image = cv2.GaussianBlur(res_image_hsv, self.grid_size, 0)

        h, s, v = cv2.split(blur_image)

        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=self.grid_size)
        clahe_s_image = clahe.apply(s)

        # Enhacement contrast image
        enhacer = ImageEnhance.Contrast(Image.fromarray(clahe_s_image))
        enhacement_contrast_s_image = np.array(enhacer.enhance(self.enhacement))

        prep_image = cv2.merge([h, enhacement_contrast_s_image, v])

        return prep_image, reescale_image, roi_color
