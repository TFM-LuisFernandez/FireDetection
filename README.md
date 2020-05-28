# FireDetection
FireDetection es un sistema destinado a complementar la motorización de incendios en situaciones de emergencia en interiores mediante técnicas de visión artificial. Este sistema está dividido en dos módulos: Firedetection Core, el algoritmo de visión, y Firedetection Web, el servidor web para realizar consultas.

# Instalación
## Dependencias para cada sistema operativo
Para la correcta ejecución del sistema son necesarias las siguientes dependencias específicas para cada sistema operativo:

- Python 3.7 (su instalación depende del sistema operativo, consultar https://www.python.org/downloads/)
- MySQL 8.0 (instalar MySQL Workbench también, consultar https://dev.mysql.com/downloads/installer/)
- Git (consultar https://gist.github.com/derhuerst/1b15ff4652a867391f03)

## Crear BBDD MySQL
Antes de poder utilizar el sistema FireDetection es imperativo crear la BBDD para el correcto funcionamiento del sistema.

![image](C:/Users/luisf/Downloads/10_2_mysql_modelo_2.png)
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
