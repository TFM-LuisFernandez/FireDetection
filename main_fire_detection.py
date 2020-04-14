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

# from numba import jit, cuda

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

UPLOAD_FOLDER = './uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
else:
    filelist = [file for file in os.listdir(UPLOAD_FOLDER)]

    if len(filelist) > 0:
        for file in filelist:
            os.remove(os.path.join(UPLOAD_FOLDER, file))

# initialize a flask object
app = Flask(__name__, template_folder='firedetection_web/templates', static_folder='firedetection_web/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def get_data():
    if request.method == "POST":
        mytype = int(request.form["mytype"])
        mysource = int(request.form["mysource"])
        myfolder = None
        myvideo = None
        mystreaming = None

        print('type', mytype)
        print('source', mysource)
        if mysource == 1:
            if request.files:
                for i in range(len(request.files)):
                    image = request.files["image" + str(i + 1)]
                    image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))

                myfolder = app.config['UPLOAD_FOLDER']

        elif mysource == 2:
            print('directorio')
            myfolder = request.form["folder"]
            print(myfolder)

        elif mysource == 3:
            print('video', request.files)
            if request.files:
                video = request.files["video"]
                video.save(os.path.join(app.config['UPLOAD_FOLDER'], video.filename))
                myvideo = app.config['UPLOAD_FOLDER'] + '/' + video.filename
                print(myvideo)

        else:
            print('TCP')
            ip = request.form['tcp_ip']
            port = request.form['tcp_port']
            mystreaming = [ip, port]

        try:
            myoutput_path = request.form["output_path"]
        except:
            myoutput_path = None
        try:
            myrecord_video = request.form["record_video"]
        except:
            myrecord_video = 1

        fire_recognizer = FireRecognizer(recognizer_type=mytype, output_path=myoutput_path)

        start_time = time.time()
        # iniciar hilo para la detección de incendios
        detection_thread = threading.Thread(target=fire_recognizer.process_path,
                                            args=(myfolder, myvideo, mystreaming))
        detection_thread.daemon = True
        detection_thread.start()
        # declarar hilo para el almacenamiento de los resultados
        record_thread = threading.Thread(target=fire_recognizer.finish,
                                         args=(myrecord_video, myoutput_path, start_time))
        record_thread.daemon = True

        return redirect(url_for('index'))

    return render_template('form.html', title='Parameters')


@app.route("/visualization")
def index():
    # devuelve la plantilla
    return render_template("index.html")


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

    def __init__(self, recognizer_type: int, output_path: str = None):
        """
        Crea una instancia del reconocedor de conjuntos de imagenes que utilizara un procesamiento concreto.

        :param recognizer_type: identificador del tipo de procesamiento a usar
        :param output_path: ruta donde se debe almacenar los resultados
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

    @app.route("/visualization")
    def recognize_video(self, video_path: str):
        """
        Realiza el reconocimiento del vídeo recibido.

        :param video_path: ruta con el video a reconocer
        :raises FileNotFoundError: si la ruta indicada no existe
        """
        input_video_path = Path(video_path)

        if input_video_path.exists():
            inputVideo = cv2.VideoCapture(video_path)
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
                        return redirect(url_for('form.html'))
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

    def recognize_image(self, image_path: str = None, frame: np.array = None):
        """
        Realiza el reconocimiento de la ruta de las imágene recibida y añade a output_images los resultados obtenidos.

        :param image_path: ruta de la imagen a reconocer
        :param frame_local: imagen a reconocer en local (paleta de falso color)
        :param frame_stream: imagen a reconocer en streaming
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
    # import time

    parser = argparse.ArgumentParser()
    # parser.add_argument("-r", "--record_video", type=int, default=1,
    #                     help='Record video from images. 0 - Yes record. 1 - No record.')
    # parser.add_argument("-o", "--output_path", type=str, help='Relative path to the output file.')
    parser.add_argument("-d", "--display", type=str, nargs=2, required=False, default=["localhost", 5000],
                        help='IP address and port for live display. <IP> <PORT>')
    # parser.add_argument("-t", "--type", type=int, required=True, help='Image type. 1 - Gray, 2 - RAINBOW')
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("-i", "--images", nargs='+', help='One or more images relative path to recognize.')
    # group.add_argument("-f", "--folder", type=str, help='Relative folder path with images to recognize.')
    # group.add_argument("-v", "--video", type=str, help='Video relative path to recognize.')
    # group.add_argument("-s", "--streaming", type=str, nargs=2,
    #                    help='IP address and port streaming video over TCP. <IP> <PORT>')
    kwargs = parser.parse_args()

    # inicio de la aplicación flask para el formulario de parámetros
    app.run(host=kwargs.display[0], port=kwargs.display[1], debug=True, threaded=True, use_reloader=False)

    # start_time = time.time()
    # # fire_recognizer = FireRecognizer(recognizer_type=kwargs.type, output_path=kwargs.output_path)
    #
    # # iniciar hilo para la detección de incendios
    # detection_thread = threading.Thread(target=fire_recognizer.process_path,
    #                                     args=(kwargs.folder, kwargs.images, kwargs.video, kwargs.streaming))
    # detection_thread.daemon = True
    # detection_thread.start()
    # # declarar hilo para el almacenamiento de los resultados
    # record_thread = threading.Thread(target=fire_recognizer.finish, args=(kwargs.record_video, kwargs.output_path))
    # record_thread.daemon = True
    #
    # # inicio de la aplicación flask para la visualización
    # app.run(host=kwargs.display[0], port=kwargs.display[1], debug=True, threaded=True, use_reloader=False)
