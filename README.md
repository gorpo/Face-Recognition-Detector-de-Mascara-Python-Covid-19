
[![Build](https://img.shields.io/badge/dev-gorpo-brightgreen.svg)]()
[![Stage](https://img.shields.io/badge/Release-Stable-brightgreen.svg)]()
[![Build](https://img.shields.io/badge/python-v3.7-blue.svg)]()
[![Build](https://img.shields.io/badge/windows-7%208%2010-blue.svg)]()
[![Build](https://img.shields.io/badge/Linux-Ubuntu%20Debian-orange.svg)]()
[![Build](https://img.shields.io/badge/arquiterura-64bits-blue.svg)]()
<h2 align="center">Detector de Mascara | Covid-19 |Face Recognition | Deep Machine Learning</h2>
<h3>Sistema de detecção facial que reconhece o uso ou não de mascaras!</h3>
<p>Este script foi desenvolvido para detecção ou não de mascaras através de reconhecimento facial, machine learning e deep learning. Esta inteligencia artificial conta com o poder de treino feito pelo usuario, ou seja, quanto mais dataset's(imagens) e fases de treino passadas pelo trainer melhor será o reconhecimento!</p>
<img src="https://github.com/gorpo/Face-Recognition-Detector-de-Mascara-Python-Covid-19/blob/master/exemplos/PYTHONES%20BANNER%20GITHUB%20INTRO.jpg" width="100%"></img>
<img src="https://github.com/gorpo/Face-Recognition-Detector-de-Mascara-Python-Covid-19/blob/master/exemplos/Screenshot_1.jpg" width="100%"></img>

# Requisitos:
- Python3.7 (não testado em outros)
- Tensorflow
- Keras
- Numpy
- Sklearn (scikit-learn)
- Imutils 
- Pillow
- h5py
- Matplotlib

# Instalações previas das libs que cumprem os requisitos para windows10:<br>
--> Tensorflow:
<code> pip install --upgrade tensorflow</code><br>
-->Keras: 
<code>pip install keras</code><br>
--> Numpy:
<code>pip install numpy scipy</code><br>
--> Scikit-learn:
<code>pip install scikit-learn</code><br>
--> Pillow:
<code>pip install pillow</code><br>
--> h5py:
<code>pip install h5py</code><br>
--> Matploitlib:
<code>pip install matplotlib</code><br>


# Dataset de imagens:
<p>Este script conta com um dataset pronto com 690 pessoas com mascaras e 686 imagens de pessoas sem mascaras, você pode melhorar este dataset ou criar seu proprio dataset. Dataset's sempre são criados com imagens "positivas" e "negativas", onde positivas são oque queremos detectar e negativas tudo q não queremos detectar, estes dataset's passam pelo trainer (trainer.py) o qual é responsável por gerar o "modelo" para nossa rede neural para o sistema de reconhecimento, portanto quanto mais imagens dentro do dataset e quantidade de treinos em cima delas melhor o desempenho do machine learnig e a execução mais precisa de nosso face recognition com atributo de máscaras.</p>

# Executando o trainer:
<p>Após ter todas as Lib's instaladas basta rodar o arquivo trainer.py, ele irá treinar as imagens da sua pasta dataset, aconselho que aumente a quantidade de imagens e a quantidade de treinos, para isto basta adicionar mais imagens nas pastas com e sem mascaras e no arquivo 'trainer.py' na linha 26 - quantidade_treinos-  definir no minimo para 30, aconselho 100x o processo.</p>

# Executando o detector de imagens:
<p>Após concluido o treino e salvo o arquivo -mask_detector.model-  na pasta /detectores você já pode iniciar suas análises de imagem, basta abrir o arquivo - detectar_mascaras_imagem.py - e na linha 19, - imagem = cv2.imread('exemplos/1.png') -  trocar a imagem pela sua e iniciar o script detector de imagens!</p>

# Executando o detector com sistema de camera:
<p>Após concluido o treino e salvo o arquivo -mask_detector.model-  na pasta /detectores você já pode iniciar suas análises de imagem da camera ou de videos, basta abrir o arquivo - detectar_mascaras_camera.py - e ele ira detectar automaticamente! Caso queira usar uma camera externa ou até mesmo seu celular como webcam usando o apk DroidCam basta ir na linha 67, - camera = 0 -  trocar o valor '0' pelo Endereço de IP de sua camera ou do Aplicativo DroidCam (ex: 'http://192.168.0.4:4747/mjpegfeed'), este adicional 'mjpegfeed' que ajuda a sua camera do celular usando DroidCam APK a ser reconhecido pelo Python. Sim este script aceita também cameras de vigilancia, basta por o IP e PORTA da camera! </p>

# Editando:
<p>Todos arquivos editaveis estão com este material, bem como todo seu código esta todo comentado para facil compreensão Aconselho uso do trainer.py em uma maquina potente como as maquinas gratuitas da Google Colab, onde você tem gratuitamente 12GB RAM, 12GB GPU, 100GB HDD e um Processador Xeon Gratuitamente além de não precisar instalar as lib's, pois elas já vem instaladas por padrão nas maquinas gratuitas disponiveis no Google Colab!</p>
    
 # Tempo de execução:
 <p>O tempo de execução de todo processo e qualidade varia de maquina para maquina, este script usa duas formas para fazer seu processo, ou uso da Memoria Ram ou uso da GPU. Para acelerar o processo aconselho uso de GPU porém ira se comportar tranquilamente com uso da memoria ram.

 
 <img src="https://github.com/gorpo/Face-Recognition-Detector-de-Mascara-Python-Covid-19/blob/master/exemplos/rodape.jpg" width="100%"></img>


