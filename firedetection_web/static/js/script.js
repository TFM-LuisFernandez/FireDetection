function source() {
    // cambio de fuente -> eliminar fuente seleccionada anteriormente si es que la habia
    if (document.getElementById('myfiles') != null) {
        $("[id='myfiles']").remove();
    } else if (document.getElementById('tcp_ip') != null) {
        $("[id='tcp_ip']").remove();
        $("[id='tcp_port']").remove();
    }

    // en funcion de la fuente seleccionada -> insertar parametros
    var x = document.getElementById("mysource").value;
    if (x == 1) {
        var newinput = document.createElement("INPUT");
        newinput.setAttribute("id", "myfiles");
        newinput.setAttribute("name", "myfiles[]");
        newinput.setAttribute("type", "file");
        newinput.setAttribute('multiple', '');
        newinput.setAttribute('autocomplete', 'off');
        newinput.setAttribute("required", "");
        newinput.setAttribute("accept", "image/*");
        document.getElementById('mysource').insertAdjacentElement('afterend', newinput);
        newinput.onchange = function () {
            var $fileUpload = $("input[type='file']");
            if (parseInt($fileUpload.get(0).files.length) % 3 !== 0) {
                alert("El total de archivos debe ser múltiplo de 3");
                $fileUpload.get(0).value = null;
            }
        }

    } else if (x == 2) {
        var newinput = document.createElement("INPUT");
        newinput.setAttribute("id", "myfiles");
        newinput.setAttribute("name", "myfiles[]");
        newinput.setAttribute("type", "file");
        newinput.setAttribute('webkitdirectory', '');
        newinput.setAttribute('mozdirectory', '');
        newinput.setAttribute('autocomplete', 'off');
        newinput.setAttribute("required", "");
        document.getElementById('mysource').insertAdjacentElement('afterend', newinput);

    } else if (x == 3) {
        var newinput = document.createElement("INPUT");
        newinput.setAttribute("id", "myfiles");
        newinput.setAttribute("name", "myfiles[]");
        newinput.setAttribute("type", "file");
        newinput.setAttribute('multiple', '');
        newinput.setAttribute('autocomplete', 'off');
        newinput.setAttribute("required", "");
        newinput.setAttribute("accept", "video/*");
        document.getElementById('mysource').insertAdjacentElement('afterend', newinput);

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

function record() {
    if (document.getElementById('output_path') == null) {
        var checkbox_rec = document.createElement('input');
        checkbox_rec.type = "checkbox";
        checkbox_rec.name = "record";
        checkbox_rec.id = "checkbox_rec";
        checkbox_rec.className = "checkbox";
        checkbox_rec.onclick = function () {
            if (document.getElementById("checkbox_rec").checked) {
                document.getElementById("checkbox_rec").value = 0;

                var checkbox_create = document.createElement('input');
                checkbox_create.type = "checkbox";
                checkbox_create.name = "create_video";
                checkbox_create.id = "checkbox_create";
                checkbox_create.className = "checkbox";
                checkbox_create.onclick = function () {
                    if (document.getElementById("checkbox_create").checked) {
                        document.getElementById("checkbox_create").value = 0;
                    } else {
                        document.getElementById("checkbox_create").value = 1;
                    }
                };
            } else {
                document.getElementById("checkbox_rec").value = 1;
                $("[id='checkbox_create']").remove();
                $("[id='label_create']").remove();
                $("[id='span_create']").remove();
                $("[id='text_create']").remove();
            }

            var label_create = document.createElement('label');
            var span_create = document.createElement('span');
            var text_create = document.createTextNode("¿Desea crear un vídeo a partir de las imágenes generadas en la detección de incendios?");
            // assigning attributes for the created label tag
            label_create.id = "label_create";
            label_create.htmlFor = "checkbox_create";
            label_create.className = "checkbox";
            text_create.id = "span_rec";
            span_create.style.color = "blue";
            text_create.id = "text_create";

            span_create.appendChild(text_create);
            label_create.appendChild(span_create);

            document.getElementById("create_video").appendChild(checkbox_create);
            document.getElementById("create_video").appendChild(label_create);

        };

        // creating label for checkbox
        var label_rec = document.createElement('label');
        var span_rec = document.createElement('span');
        var text_rec = document.createTextNode("¿Desea almacenar los resultados obtenidos en la detección de incendios?");
        // assigning attributes for the created label tag
        label_rec.id = "label_rec";
        label_rec.htmlFor = "checkbox_rec";
        label_rec.className = "checkbox";
        text_rec.id = "span_rec";
        span_rec.style.color = "blue";
        text_rec.id = "text_rec";

        span_rec.appendChild(text_rec);
        label_rec.appendChild(span_rec);

        document.getElementById("record").appendChild(checkbox_rec);
        document.getElementById("record").appendChild(label_rec);
    }
}

function remove_record() {
    if (document.getElementById('output_path') != null) {
        // $("[id^='output_path']").remove();
        $("[id^='checkbox']").remove();
        $("[id^='label']").remove();
        $("[id^='span']").remove();
        $("[id^='text']").remove();
    }
}

function restart() {
    if (document.getElementById('myfiles') != null) {
        $("[id='myfiles']").remove();
    } else if (document.getElementById('tcp_ip') != null) {
        $("[id='tcp_ip']").remove();
        $("[id='tcp_port']").remove();
    }

    if (document.getElementById('output_path') != null) {
        // $("[id^='output_path']").remove();
        $("[id^='checkbox']").remove();
        $("[id^='label']").remove();
        $("[id^='span']").remove();
        $("[id^='text']").remove();
    }
}