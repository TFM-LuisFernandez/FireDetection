class FireClassifier:
    """Clasificador de las regiones segmentadas en la imagen térmica."""

    def __init__(self, shape_threshold_min: float, shape_threshold_max: float,
                 area_threshold: int, mean_threshold: float):
        """
        Crea un clasificador de regiones que son fuego.

        :param shape_threshold_min: umbral inferior para determinar si la forma del objeto es fuego o no
        :param shape_threshold_max: umbral superior para determinar si la forma del objeto es fuego o no
        :param area_threshold: umbral para descartar pequeñas regiones que no son de interés
        :param mean_threshold: umbral para determinar si la media del color de la región es fuego
        """
        self.shape_threshold_min = shape_threshold_min
        self.shape_threshold_max = shape_threshold_max
        self.area_threshold = area_threshold
        self.mean_threshold = mean_threshold

    def classifying_fire(self, features: dict) -> dict:
        """
        Reconoce las características de las regiones segmentadas.

        :param features: diccionario con las características de cada región candidata de la imagen actual
        :returns: diccionario de parámetros de als regiones que son clasifiacadas como fuego
        """
        fire_features = {'contours': list(), 'bd': list(), 'area': list(), 'centroids': list(), 'mean': list()}

        for object in range(len(features['contours'])):
            if (features['bd'][object] > self.shape_threshold_min) and \
                    (features['bd'][object] < self.shape_threshold_max) and \
                    (features['area'][object] > self.area_threshold) and \
                    (features['mean'][object] > self.mean_threshold):

                for key in features.keys():
                    fire_features[key].append(features[key][object])

        return fire_features
