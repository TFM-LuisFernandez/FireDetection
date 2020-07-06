# FireDetection
FireDetection es un sistema destinado a complementar la motorización de incendios en situaciones de emergencia en interiores mediante técnicas de visión artificial. Este sistema está dividido en dos módulos: Firedetection Core, el algoritmo de visión, y Firedetection Web, el servidor web para realizar consultas.

# Instalación
## Dependencias para cada sistema operativo
Para la correcta ejecución del sistema son necesarias las siguientes dependencias específicas para cada sistema operativo:

- Python 3.7 (su instalación depende del sistema operativo, consultar https://www.python.org/downloads/)
- MySQL 8.0 (consultar https://dev.mysql.com/downloads/installer/)
- MySQL Workbench (consultar https://dev.mysql.com/downloads/workbench/)
- LabVIEW (consultar https://www.ni.com/es-es/support/downloads/software-products/download.labview.html#346254)
- Git (consultar https://gist.github.com/derhuerst/1b15ff4652a867391f03)

## Crear BBDD MySQL
Antes de poder utilizar el sistema FireDetection es imperativo crear la BBDD para el correcto funcionamiento del sistema. Para ello hay que hacer los siguientes pasos:
```
1) Crear en MySQL Workbench una nueva conexión con el usuario "FireDetection" con la contraseña "firedetection".
2) Acceder a la nueva conexión creada y crear la BBDD "firedetection_bbdd".
3) Crear 2 tablas: "imagenes" y "parametros".
4) Campos de la tabla imagenes: id(int), name(varchar(50)), result(longblob), mask(longblob), time(datetime), detection(varchar(20).
5) Campos de la tabla parametros: id(int), area(int), bd(float), cx(int), cy(int), imagenes_id(int)
6) Establecer una conexión 1(imagenes):n(parametros)
```

## Descarga del sistema e instalación de dependencias de Python (comunes a todos los sistema operativos)
Una vez resueltas las dependencias anteriores, para descargar el sistema se debe ejecutar:

```
https://github.com/TFM-LuisFernandez/FireDetection.git
```
Entramos en el directorio generado
```
cd FireDetection
```
Los paquetes de Python necesarios se deben instalar de la siguiente manera:
```
pip install -r requirements.txt
```
Con esto habría finalizado la instalación y ya se podría usar el sistema.

# Ejecución
Para ejecitar el software de adquisición de imágenes térmicas de LabVIEW, se debe abrir el fichero "Labview_FLIR_A65_TCP.vi", establecer los parámetros de configuración como se indica a continuación:
  * FrameRate: NTSC30HZ
  * IRFormat: HighGainMode
  * Ring: Mono 8
  ---
  **NOTE**

  Estos parámetros son los adecuados para la cámara FLIR A65

  ---
Para ejecutar el sistema FireDetection se debe ejecutar el siguiente comando desde la raíz del repositorio:
```
start.bat
```
También se podría ejecutar haciendo doble click sobre el fichero **start.bat**.

Una tercera opción sería invocarlo desde un terminal, accediendo al directorio del ssistema FireDetection, proporcionando una dirección IP y Puerto como se muestra acontinuación:
```
python main_fire_detection.py -d <IP> <PUERTO>
```

Una vez iniciado el contenedor Docker hay que navegar a la siguiente dirección para usar el sistema: http://127.0.0.1:5000/index.html
