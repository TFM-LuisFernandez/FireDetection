from firedetection_core.fire_recognition import Recognizer
from firedetection_core.fire_visualization import FireVisualize
from firedetection_core.fire_record import FireRecord

import socket
import threading
import cv2
import os
import numpy as np
import time

from PIL import Image
from pathlib import Path
from typing import List
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

# from numba import jit, cuda

# Inicializar el frame de salida (outputFrame) y un hilo (lock) usado para asegurar el intercambio de los frames
# de salida (útil para múltiples navegadores/pestañas que estén viendo la visualizacion)
outputFrame = None
lock = threading.Lock()
# Definimos directorios de subida de ficheros y de almacenamiento de resultados
UPLOAD_FOLDER = './uploads'
RESULT_FOLDER = './results'
# Definimos las extensiones permitidas de los archivos recibidos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'mpg', 'mpeg', 'ogv', 'mov', 'mp4', 'avi'}
# Comprobamos que el directorio de subida está creado y vacío
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
else:
    filelist = [file for file in os.listdir(UPLOAD_FOLDER)]

    if len(filelist) > 0:
        for file in filelist:
            os.remove(os.path.join(UPLOAD_FOLDER, file))

# Inicializamos el objeto Flask
app = Flask(__name__, template_folder='firedetection_web/templates', static_folder='firedetection_web/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    """
    Determina si el archivo recibido tiene una extension de imagen o video
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/index.html", methods=["GET", "POST"])
def get_data():
    """
    Recibe del formulario, mediante POST, los parametros del algoritmo junto con los ficheros de subida
    """
    if request.method == "POST":
        # Recibimos el tipo de imagenes y el tipo de fuente
        mytype = int(request.form["mytype"])
        mysource = int(request.form["mysource"])
        myfolder = None
        myvideos = None
        mystreaming = None

        if 1 <= mysource <= 3:
            # Si son ficheros locales, se almacenan en el directorio uploads
            files = request.files.getlist("myfiles[]")

            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            if mysource <= 2:
                myfolder = app.config['UPLOAD_FOLDER']
            elif mysource == 3:
                myvideos = app.config['UPLOAD_FOLDER']

        else:
            # Si es una conexion TCP en streaming, solo se transmite la IP y el puerto (no se almacena nada)
            print('TCP')
            ip = request.form['tcp_ip']
            port = request.form['tcp_port']
            mystreaming = [ip, port]

        # Comprobamos los parámetros opcionales
        try:
            # Checkbox para almacenar resultados
            myrecord = int(request.form["record"])
            if myrecord == 0:
                # Checkbox para guardar resultados está seleccionado -> comprobar directorio results
                myoutput_path = RESULT_FOLDER

                if not os.path.exists(RESULT_FOLDER):
                    os.makedirs(RESULT_FOLDER)
                else:
                    filelist = [file for file in os.listdir(RESULT_FOLDER)]

                    if len(filelist) > 0:
                        for file in filelist:
                            os.remove(os.path.join(RESULT_FOLDER, file))
        except:
            myoutput_path = None

        try:
            # Comprobar checkbox para crear vídeo a partir de las imagenes generadas por el algoritmo
            myrecord_video = request.form["create_video"]
        except:
            myrecord_video = 1

        start_time = time.time()
        fire_recognizer = FireRecognizer(recognizer_type=mytype, output_path=myoutput_path, create_video=myrecord_video,
                                         start_time=start_time)
        # iniciar hilo para la detección de incendios
        detection_thread = threading.Thread(target=fire_recognizer.process_path,
                                            args=(myfolder, myvideos, mystreaming))
        detection_thread.daemon = True
        detection_thread.start()
        # declarar hilo para el almacenamiento de los resultados
        # record_thread = threading.Thread(target=fire_recognizer.finish,
        #                                  args=(myrecord_video, myoutput_path, start_time))
        # record_thread.daemon = True

        return redirect(url_for('visual'))

    return render_template('form.html', title='Parameters')


@app.route("/visualization.html")
def visual():
    # devuelve la plantilla
    return render_template("visual.html")


@app.route("/video_feed")
def video_feed():
    # devuelve la respuesta generada -> Frame resultante
    # tipo (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def generate():
    # coge las referencias globales
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


class FireRecognizer:
    """Reconocedor de fuego en imagen térmica."""

    def __init__(self, recognizer_type: int, output_path: str = None, create_video: int = 1, start_time: float = None):
        """
        Crea una instancia del reconocedor de conjuntos de imagenes que utilizara un procesamiento concreto.

        :param recognizer_type: identificador del tipo de procesamiento a usar
        :param output_path: ruta donde se debe almacenar los resultados
        :param create_video: identificador para crear o no vídeo con las imágenes generadas por el algoritmo
        :param start_time: tiempo de inicio de ejecucion del algoritmo
        :raises NotADirectoryError: si la ruta indicada no es un directorio
        :raises FileNotFoundError: si la ruta indicada no existe
        """
        self.recognizer = Recognizer(recognizer_type=recognizer_type)
        self.visualizer = FireVisualize()
        self.output_path = output_path

        if output_path is not None:
            output_dataset_path = Path(output_path)

            if output_dataset_path.exists() and output_dataset_path.is_dir():
                self.record = FireRecord(file_json=open(output_path + '/results_fire_detection.json', 'w',
                                                        encoding='utf-8'), path=output_path)
            elif not output_dataset_path.exists():
                raise FileNotFoundError('Specified data output_path does not exist: ' + str(output_dataset_path))
            else:
                raise NotADirectoryError('Specified data output_path is not a directory:  ' + str(output_dataset_path))

        self.create_video = create_video
        self.start_time = start_time
        self.buffer_images = list()
        self.total_images = 0

    def process_path(self, folder_path: str = None, video_path: str = None, streaming_params: list = None):
        """
        Realiza el reconocimiento de todas las imágenes presentes en una carpeta o de todas las rutas a imágenes recibida.

        :param folder_path: ruta relativa de la carpeta con imágenes a reconocer
        :param video_path: ruta relativa del video a reconocer
        :param streaming_params: IP y puerto de la conexión TCP
        :raises TypeError: si la ruta indicada no existe
        """
        if folder_path is not None:
            self.recognize_folder(folder_path)
        elif video_path is not None:
            self.recognize_video(video_path)
        elif streaming_params is not None:
            self.recognize_streaming(streaming_params)
        else:
            raise TypeError('No args have been passed')

    def recognize_folder(self, folder_path: str):
        """
        Realiza el reconocimiento de todas las imágenes presentes en una carpeta.

        :param folder_path: ruta relativa de la carpeta con imágenes a reconocer
        :raises NotADirectoryError: si la ruta indicada no es un directorio
        :raises FileNotFoundError: si la ruta indicada no existe
        """
        input_dataset_path = Path(folder_path)

        if input_dataset_path.exists() and input_dataset_path.is_dir():
            dataset_images_path = [str(img_path) for img_path in sorted(input_dataset_path.glob('*.*'),
                                                                        key=os.path.getmtime)]
            self.recognize_images(dataset_images_path)
        elif not input_dataset_path.exists():
            raise FileNotFoundError('Specified data path does not exist: ' + str(input_dataset_path))
        else:
            raise NotADirectoryError('Specified data path is not a directory:  ' + str(input_dataset_path))

    # @app.route("/visualization")
    def recognize_video(self, video_path: str):
        """
        Realiza el reconocimiento del vídeo recibido.

        :param video_path: ruta con el video a reconocer
        :raises FileNotFoundError: si la ruta indicada no existe
        """
        input_video_path = Path(video_path)

        if input_video_path.exists() and input_video_path.is_dir():
            dataset_video_path = [str(img_path) for img_path in sorted(input_video_path.glob('*.*'),
                                                                       key=os.path.getmtime)]
            input_paths = [Path(path) for path in dataset_video_path]

            for number_video, input_video_path in enumerate(input_paths):
                if input_video_path.exists():
                    inputVideo = cv2.VideoCapture(dataset_video_path[number_video])
                    if not inputVideo.isOpened():
                        print("Error opening video stream or file")
                    else:
                        # width = int(inputVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
                        # height = int(inputVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print("Total number of frames: %d frames" % (inputVideo.get(cv2.CAP_PROP_FRAME_COUNT)))
                        print("FPS: %d" % (inputVideo.get(cv2.CAP_PROP_FPS)))
                        minutes = int((inputVideo.get(cv2.CAP_PROP_FRAME_COUNT) / inputVideo.get(cv2.CAP_PROP_FPS)) / 60)
                        seconds = int(inputVideo.get(cv2.CAP_PROP_FRAME_COUNT) / inputVideo.get(cv2.CAP_PROP_FPS)) % 60
                        print("Duration of video: " + str(minutes) + " minutes : " + str(seconds) + " seconds")

                        while inputVideo.isOpened():
                            ret, frame = inputVideo.read()
                            if ret:
                                self.recognize_image(frame=frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                            else:
                                # return redirect(url_for('form.html'))
                                self.finish(record=self.create_video, path=self.output_path, start_time=self.start_time)
                                break

                        inputVideo.release()

                elif not input_video_path.exists():
                    raise FileNotFoundError('Specified data video path does not exist: ' + str(input_video_path))

    def recognize_streaming(self, params: list):
        """
        Realiza el reconocimiento de las imágenes recibidas en Streaming vía TCP.

        :param params: Dirección y Puerto desde donde se reciben las imágenes a reconocer
        :raises ConnectionRefusedError:  no se puede establecer una conexión con los parámetros establecidos
        """
        TCP_IP = params[0]
        TCP_PORT = int(params[1])
        BUFFER_IMAGE = 640 * 512  # height * width * 2 -> 16 bits

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((TCP_IP, TCP_PORT))

            while True:
                bits_image = b''
                size_bits = len(bits_image)
                while len(bits_image) < BUFFER_IMAGE:
                    bits_image += s.recv(2 ** 20)
                    if size_bits == len(bits_image):
                        self.finish(record=self.create_video, path=self.output_path, start_time=self.start_time)
                        raise ConnectionError('No client data received')

                frame = np.array(Image.frombytes("L", (640, 512), bits_image))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                self.recognize_image(frame=frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            s.close()
        except:
            raise ConnectionRefusedError('Cannot establish TCP connection with %s address and port %d'
                                         % (TCP_IP, TCP_PORT))

    def recognize_images(self, images_path: List[str]):
        """
        Realiza el reconocimiento de todas las rutas de las imágenes recibidas.

        :param images_path: lista con las rutas de las imágenes a reconocer
        :raises NotAFileError: si alguna de las rutas indicadas no es un fichero
        :raises FileNotFoundError: si alguna de las rutas indicadas no existe
        :raise TypeError: si el argumento recibido no es una lista
        """
        if isinstance(images_path, list):
            input_paths = [Path(path) for path in images_path]
            print('Total number of images: %d images' % (len(input_paths)))
        else:
            raise TypeError('Image(s) path must be a list.')

        for path in input_paths:
            self.recognize_image(image_path=path)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.finish(record=self.create_video, path=self.output_path, start_time=self.start_time)

    def recognize_image(self, image_path: str = None, frame: np.array = None):
        """
        Realiza el reconocimiento de la ruta de las imágene recibida y añade a output_images los resultados obtenidos.

        :param image_path: ruta de la imagen a reconocer
        :param frame: imagen a reconocer procedente de un video o una conexión TCP
        :raises FileNotFoundError: si la ruta indicada no existe
        """
        global outputFrame, lock

        if image_path is not None:
            image_path = Path(image_path)

            if image_path.is_file():
                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            elif not image_path.exists():
                raise FileNotFoundError('Specified image path does not exist: ' + str(image_path))

        # st = time.time()
        self.buffer_images.append(frame)
        if len(self.buffer_images) == 3:
            # cv2.imshow('buffer_entrada', cv2.hconcat([self.buffer_images[0], self.buffer_images[1], self.buffer_images[2]]))
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            fire_results = self.recognizer.recognize_image(buffer=self.buffer_images)

            # et = time.time()
            # print(et - st)  # tiempo procesamiento 1 buffer
            result_image, mask_image = self.visualizer.visualize_restult(image=fire_results['roi'],
                                                                         contours=fire_results['features']['contours'],
                                                                         centroids=fire_results['features']
                                                                                               ['centroids'])

            with lock:
                outputFrame = cv2.hconcat([result_image, mask_image])

            if self.output_path is not None:
                json_file = {'identifier': str(self.total_images) + '.jpg',
                             'detection': fire_results['detection'],
                             'time': datetime.now().__str__(),
                             'area': fire_results['features']['area'],
                             'bd': fire_results['features']['bd'],
                             'centroids': fire_results['features']['centroids']}
                self.record.record_output_image(images=[result_image, mask_image, self.buffer_images],
                                                json_file=json_file)

            self.buffer_images.clear()
            self.total_images += 1

    def finish(self, record: int = 1, path: str = None, start_time: float = None):
        print('--- Processed images: %d images ---' % self.total_images),
        print('--- Processing time: ' + str(int((time.time() - start_time) / 60)) + ' minutes : ' +
              str(int((time.time() - start_time) % 60)) + ' seconds ---')

        if record == 0 and path is not None:
            print('--- CREATING VIDEO FROM THE STORED IMAGES  --- PLEASE WAIT UNTIL THE PROCESS IS OVER')
            self.record.record_output_video(path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--display", type=str, nargs=2, required=False, default=["localhost", 5000],
                        help='IP address and port for live display. <IP> <PORT>')

    kwargs = parser.parse_args()

    # inicio de la aplicación flask para el formulario de parámetros
    app.run(host=kwargs.display[0], port=kwargs.display[1], debug=True, threaded=True, use_reloader=False)
