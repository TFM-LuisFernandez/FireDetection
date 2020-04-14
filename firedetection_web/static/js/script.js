function source() {
    // cambio de fuente -> eliminar fuente seleccionada anteriormente si es que la habia
    if (document.getElementById('image1') != null) {
        $("[id^='image']").remove();
        $("[id='add']").remove();
    } else if (document.getElementById('folder') != null) {
        $("[id='folder']").remove();
    } else if (document.getElementById('video') != null) {
        $("[id^='video']").remove();
        $("[id='add']").remove();
    } else if (document.getElementById('tcp_ip') != null) {
        $("[id='tcp_ip']").remove();
        $("[id='tcp_port']").remove();
    }

    // en funcion de la fuente seleccionada -> insertar parametros
    var x = document.getElementById("mysource").value;
    if (x == 1) {
        var image_number = 1;
        newimage(image_number)

    } else if (x == 2) {
        // var newinput = document.createElement("INPUT");
        // newinput.setAttribute("id", "myfiles");
        // newinput.setAttribute("name", "myfiles[]");
        // newinput.setAttribute("type", "file");
        // newinput.setAttribute('webkitdirectory', '');
        // newinput.setAttribute('mozdirectory', '');
        // newinput.setAttribute('autocomplete', 'off');
        // newinput.setAttribute("required", "");
        // document.getElementById('mysource').insertAdjacentElement('afterend', newinput);
        // newinput.onchange = function () {
        //     var fileInput = document.getElementById('myfiles');
        //     var fileList = [];
        // }

        var newinput = document.createElement("INPUT");
        newinput.setAttribute("id", "folder");
        newinput.setAttribute("name", "folder");
        newinput.setAttribute("placeholder", "Introduce la ruta del directorio de las imágenes");
        newinput.setAttribute("type", "text");
        newinput.setAttribute("required", "");
        document.getElementById('mysource').insertAdjacentElement('afterend', newinput);


    } else if (x == 3) {
        var video_number = 1;
        newvideo(video_number)

        // var newinput = document.createElement("INPUT");
        // newinput.setAttribute("id", "video");
        // newinput.setAttribute("name", "video");
        // newinput.setAttribute("type", "file");
        // newinput.setAttribute("required", "");
        // newinput.setAttribute("accept", "video/*");
        // document.getElementById('mysource').insertAdjacentElement('afterend', newinput);
    } else {
        var newinput_port = document.createElement("INPUT");
        newinput_port.setAttribute("id", "tcp_port");
        newinput_port.setAttribute("name", "tcp_port");
        newinput_port.setAttribute("type", "number");
        newinput_port.setAttribute("required", "");
        newinput_port.setAttribute("minlength", "4");
        newinput_port.setAttribute("maxlength", "5");
        newinput_port.setAttribute("placeholder", "Puerto");
        document.getElementById('mysource').insertAdjacentElement('afterend', newinput_port);

        var newinput_ip = document.createElement("INPUT");
        newinput_ip.setAttribute("id", "tcp_ip");
        newinput_ip.setAttribute("name", "tcp_ip");
        newinput_ip.setAttribute("type", "text");
        newinput_ip.setAttribute("required", "");
        newinput_ip.setAttribute("minlength", "7");
        newinput_ip.setAttribute("maxlength", "15");
        newinput_ip.setAttribute("pattern", "((^|\\.)((25[0-5])|(2[0-4]\\d)|(1\\d\\d)|([1-9]?\\d))){4}$");
        newinput_ip.setAttribute("placeholder", "Dirección IP");
        document.getElementById('mysource').insertAdjacentElement('afterend', newinput_ip);
    }
}

function newimage(image_number) {
    for (var i = 0; i < 3; i++) {
        var newinput = document.createElement("INPUT");
        newinput.setAttribute("id", "image" + image_number);
        newinput.setAttribute("name", "image" + image_number);
        newinput.setAttribute("type", "file");
        newinput.setAttribute("required", "");
        newinput.setAttribute("accept", "image/*");

        if (image_number === 1) {
            document.getElementById('mysource').insertAdjacentElement('afterend', newinput);
        } else {
            document.getElementById('image' + (image_number - 1)).insertAdjacentElement('afterend', newinput);
        }

        image_number += 1;
    }

    var newbutton_add = document.createElement("BUTTON");
    newbutton_add.setAttribute("id", "add");
    newbutton_add.innerHTML = "Añadir 3 imágenes";
    document.getElementById('image' + (image_number - 1)).insertAdjacentElement('afterend', newbutton_add);
    newbutton_add.onclick = function () {
        newbutton_add.remove();
        newimage(image_number);
    }
}

function newvideo(video_number) {
    var newinput = document.createElement("INPUT");
    newinput.setAttribute("id", "video" + video_number);
    newinput.setAttribute("name", "video" + video_number);
    newinput.setAttribute("type", "file");
    newinput.setAttribute("required", "");
    newinput.setAttribute("accept", "video/*");

    if (video_number === 1) {
        document.getElementById('mysource').insertAdjacentElement('afterend', newinput);
    } else {
        document.getElementById('video' + (video_number - 1)).insertAdjacentElement('afterend', newinput);
    }

    var newbutton_add = document.createElement("BUTTON");
    newbutton_add.setAttribute("id", "add");
    newbutton_add.innerHTML = "Añadir otro vídeo";
    document.getElementById('video' + video_number).insertAdjacentElement('afterend', newbutton_add);
    newbutton_add.onclick = function () {
        video_number += 1;
        newbutton_add.remove();
        newvideo(video_number);
    }
}

function record() {
    if (document.getElementById('output_path') == null) {
        var myDiv_path = document.getElementById("path");
        var myDiv_rec = document.getElementById("record");

        var newinput = document.createElement("INPUT");
        newinput.setAttribute("id", "output_path");
        newinput.setAttribute("name", "output_path");
        newinput.setAttribute("placeholder", "Introduce la ruta del directorio donde se van a almacenar los resultados");
        newinput.setAttribute("type", "text");
        newinput.setAttribute("required", "");

        myDiv_path.appendChild(newinput);

        // creating checkbox element
        var checkbox = document.createElement('input');
        // Assigning the attributes
        // to created checkbox
        checkbox.type = "checkbox";
        checkbox.name = "record_video";
        checkbox.id = "checkbox_1";
        checkbox.className = "checkbox";
        checkbox.onclick = function () {
            if (document.getElementById("checkbox_1").checked) {
                document.getElementById("checkbox_1").value = 0;
            } else {
                document.getElementById("checkbox_1").value = 1;
            }

        };


        // creating label for checkbox
        var label = document.createElement('label');
        var span = document.createElement('span');
        var text = document.createTextNode("¿Desea crear un vídeo a partir de las imágenes generadas en la detección de incendios?");
        // assigning attributes for the created label tag
        label.id = "label";
        label.htmlFor = "checkbox_1";
        label.className = "checkbox";
        span.id = "span";
        span.style.color = "blue";
        text.id = "text";
        // appending the created text to the created label tag
        // label.appendChild(document.createTextNode('¿Desea crear un vídeo a partir de las imágenes generadas en la detección de incendios?'));
        span.appendChild(text);
        label.appendChild(span);
        // appending the checkbox
        // and label to div
        myDiv_rec.appendChild(checkbox);
        myDiv_rec.appendChild(label);
    }
}

function remove_record() {
    if (document.getElementById('output_path') != null) {
        $("[id^='output_path']").remove();
        $("[id^='checkbox_1']").remove();
        $("[id^='label']").remove();
        $("[id^='span']").remove();
        $("[id^='text']").remove();
    }
}

function restart() {
    if (document.getElementById('image1') != null) {
        $("[id^='image']").remove();
        $("[id='add']").remove();
    } else if (document.getElementById('folder') != null) {
        $("[id='folder']").remove();
    } else if (document.getElementById('video') != null) {
        $("[id^='video']").remove();
        $("[id='add']").remove();
    } else if (document.getElementById('tcp_ip') != null) {
        $("[id='tcp_ip']").remove();
        $("[id='tcp_port']").remove();
    }

    if (document.getElementById('output_path') != null) {
        $("[id^='output_path']").remove();
        $("[id^='checkbox_1']").remove();
        $("[id^='label']").remove();
        $("[id^='span']").remove();
        $("[id^='text']").remove();
    }
}