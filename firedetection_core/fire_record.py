from pathlib import Path

import cv2
import os
import json


class FireRecord:
    """Grabador de resultados en un directorio"""

    def __init__(self, file_json, path: str = None):
        """Crea un grabador de imágenes y fichero json"""
        self.file_json = file_json
        self.path = path
        self.iter = 0

    def record_output_image(self, images: list, json_file: dict):
        """
        Almacenamiento de las imágenes resultantes del reconocimiento de las regiones de fuego

        :param images: imagenes con las regiones donde hay fuego reconocidas
        :param json_file: diccionario con las características del reconocimiento del fuego
        :raises NotADirectoryError: si la ruta indicada no es un directorio
        :raises FileNotFoundError: si la ruta indicada no existe
        """
        if self.path is not None:
            output_dataset_path = Path(self.path)

            if output_dataset_path.exists() and output_dataset_path.is_dir():
                if not os.path.exists(self.path + '/resultados/'):
                    os.makedirs(self.path + '/resultados/')
                    os.makedirs(self.path + '/originales/')

                cv2.imwrite(os.path.join(self.path + '/resultados/', json_file['identifier']),
                            cv2.hconcat([images[0], images[1]]))

                for image in images[2]:
                    cv2.imwrite(os.path.join(self.path + '/originales/', str(self.iter) + '.jpg'), image)
                    self.iter += 1

                json.dump(json_file, self.file_json, indent=4)


            elif not output_dataset_path.exists():
                raise FileNotFoundError('Specified data path does not exist: ' + str(output_dataset_path))
            else:
                raise NotADirectoryError('Specified data path is not a directory:  ' + str(output_dataset_path))

    def record_output_video(self, folder_path):
        """
        Realiza el reconocimiento de todas las imágenes presentes en una carpeta

        :param folder_path: ruta relativa de la carpeta con imágenes para crear el vídeo
        :raises NotADirectoryError: si la ruta indicada no es un directorio
        :raises FileNotFoundError: si la ruta indicada no existe
        """
        if folder_path is not None:
            folder_dataset_path = Path(folder_path + '/resultados')
            video_output_path = folder_path + '/video/'

            if folder_dataset_path.exists() and folder_dataset_path.is_dir():
                dataset_images_path = [str(img_path) for img_path in
                                       sorted(folder_dataset_path.glob('*.jpg'), key=os.path.getmtime)]
                number = 1

                if not os.path.exists(video_output_path):
                    os.makedirs(video_output_path)

                output_dataset_path = Path(video_output_path)
                if output_dataset_path.exists() and output_dataset_path.is_dir():
                    while len(dataset_images_path) != 0:
                        iter = 0
                        img_array = list()
                        for filename in dataset_images_path:
                            img = cv2.imread(filename)
                            img_array.append(img)
                            iter += 1
                            if iter == 1000:
                                del dataset_images_path[:1000]
                                break
                            if len(img_array) == len(dataset_images_path):
                                dataset_images_path.clear()
                                break

                        height, width, layers = img_array[0].shape
                        size = (width, height)
                        out = cv2.VideoWriter((video_output_path + str(number) + 'project.avi'),
                                              cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

                        for i in range(len(img_array)):
                            out.write(img_array[i])
                        out.release()
                        number += 1

                    print('end record videos')

            elif not folder_dataset_path.exists():
                raise FileNotFoundError('Specified data path does not exist: ' + str(folder_dataset_path))
            else:
                raise NotADirectoryError('Specified data path is not a directory:  ' + str(folder_dataset_path))
