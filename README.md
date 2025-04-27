Requirements:

    - python 3.10
    - tensorflow 2.10
    - CUDA 11.2
    - cuDNN 8.1.0

Installation Guide:

    1. Install python 3.10 and add it to PATH. (preparation for tensorflow with gpu on native-Windows)
    2. clone this repository and open it in the IDE
    3. create new virtualenvironment for python 3.10 (add new Interpreter-->VirtualEnvironment)
    4. activate the new venv with ".\.venv\Scripts\activate"
    5. pip install tensorflow==2.10 (last version of tensorflow with gpu support on native-Windows)


CUDA 11.2 installation:

    Deinstalliere dein aktuelles CUDA (12.6):

        Gehe zu Systemsteuerung > Programme und Features.

        Suche nach NVIDIA CUDA Toolkit 12.6 und deinstalliere es.

        Lass den NVIDIA Treiber unangetastet! Nur das CUDA Toolkit entfernen.

    CUDA 11.2 herunterladen:

    → Hier der Link:
    CUDA 11.2.2 Archive Download
    https://developer.nvidia.com/cuda-11.2.2-download-archive

    Installation:

        Lade das "exe (local)" Installationspaket herunter (nicht network!).

        Installiere CUDA 11.2 ganz normal. Der Installer fragt dich eventuell nach den Treibern — nur Toolkit installieren, nicht die Treiber überschreiben!

cuDNN 8.1.0 installation:

    Download:

    cuDNN 8.1.0 bekommst du hier:
    cuDNN Archive Download 8.1.0
    https://developer.nvidia.com/rdp/cudnn-archive#a-collapse811-122
    (Download cuDNN v8.1.1 (Feburary 26th, 2021), for CUDA 11.0,11.1 and 11.2)

    (Du musst ein kostenloses NVIDIA Developer-Konto haben und eingeloggt sein.)

    Entpacken:

    Lade die Windows-Version (cuDNN v8.1.0 (February 26th, 2021), for CUDA 11.2) herunter und entpacke sie.

    Manuell kopieren:

    Kopiere dann die Dateien in dein CUDA 11.2 Verzeichnis:

        Inhalt von cuda\bin\* → in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin

        Inhalt von cuda\include\* → in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include

        Inhalt von cuda\lib\x64\* → in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64

    (Einfach Dateien rüberkopieren und vorhandene überschreiben, falls gefragt.)
