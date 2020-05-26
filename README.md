
[![Build](https://img.shields.io/badge/dev-gorpo-brightgreen.svg)]()
[![Stage](https://img.shields.io/badge/Release-Stable-brightgreen.svg)]()
[![Build](https://img.shields.io/badge/python-v3.7-blue.svg)]()
[![Build](https://img.shields.io/badge/windows-7%208%2010-blue.svg)]()
[![Build](https://img.shields.io/badge/Linux-Ubuntu%20Debian-orange.svg)]()
[![Build](https://img.shields.io/badge/arquiterura-64bits-blue.svg)]()
  <h6 align="center"> <h2 align="center">Detector de Mascara | Covid-19 |Face Recognition | Deep Machine Learning</h2> </h6>
<img src="https://github.com/gorpo/Face-Recognition-Detector-de-Mascara-Python-Covid-19/blob/master/exemplos/Screenshot_1.jpg?raw=true" width="100%"></img>
<h3>Sistema de detecção facial que reconhece o uso ou não de mascaras!</h3><br>
<p>Este script foi desenvolvido para detecção ou não de mascaras através de reconhecimento facial, machine learning e deep learning. Esta inteligencia artificial conta com o poder de treino feito pelo usuario, ou seja, quanto mais dataset's(imagens) e fases de treino passadas pelo trainer melhor será o reconhecimento!</p>

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
--> Tensorflow:<br>
<code> pip install --upgrade tensorflow</code><br>
-->Keras: <br>
<code>pip install keras</code><br>
--> Numpy<br>
<code>pip install numpy scipy</code><br>
--> Scikit-learn<br>
<code>pip install scikit-learn</code><br>
--> Numpy<br>
<code>pip install numpy</code><br>
--> Pillow<br>
<code>pip install pillow</code><br>
--> h5py<br>
<code>pip install h5py</code><br>
--> Matploitlib<br>
<code>pip install matplotlib</code><br>



# Executando:
<p>Após ter todas as Lib's instaladas basta rodar o arquivo trainer.py, ele irá treinar as imagens da sua pasta dataset, aconselho que aumente a quantidade de imagens e a quantidade de treinos, para isto basta adicionar mais imagens nas pastas com e sem mascaras e no arquivo 'trainer.py' na linha 26 - quantidade_treinos-  definir no minimo para 30, aconselho 100x o processo.</p>


# Editando:
<p>Todos arquivos editaveis estão com este material, dentre eles o arquivo mainwindow.ui para ser editado no Qt Design e arquivos photoshop com imagens que foram usadas neste projeto. Todas funções foram colocadas em arquivos separados para facil compreensão e o widget central que mostra as telas chama-se: stackedWidget.<br>As cores e estilos foram feitos todos em CSS dentro do arquivo mainwindow.ui do QT Design mas podem ser alterados no arquivo mainwindow.py tranquilamente!</p>
 # Comandos para serem executados no teminal ou cmd para gerar os arquivos python feitos no QT Design:
 <p>criar arquivo mainwindow.py:<br>
    <code>pyuic5 -x mainwindow.ui -o mainwindow.py</code><br>
criar arquivo files_rc_rc.py<br>
    <code>pyrcc5 -o files_rc_rc.py files_rc.qrc</code></p>
    
 # Tempo de execução:
 <p>O tempo de execução de todo processo e qualidade varia de maquina para maquina, este script usa duas formas para fazer seue processo, ou uso da Memoria Ram ou uso da GPU. Para acelerar o processo aconselho uso de GPU porém ira se comportar tranquilamente com uso da memoria ram.

 # Demonstração do layout e algumas funções:
 <h2 align="center">PAINEL NORMAL</h2>
<img src="https://raw.githubusercontent.com/gorpo/PyQt5-Modern-Interface-/master/images/examples/01.jpg" width="100%"></img>
 <h2 align="center">MENU ESQUERDA E SUBMENUS COM SUBMENUS</h2>
<img src="https://raw.githubusercontent.com/gorpo/PyQt5-Modern-Interface-/master/images/examples/02.jpg" width="100%"></img>
 <h2 align="center">"MENU INICIAR COM SUBMENUS"</h2>
<img src="https://raw.githubusercontent.com/gorpo/PyQt5-Modern-Interface-/master/images/examples/03.jpg" width="100%"></img>
 <h2 align="center">WEB BROWSER INTEGRADO FEITO EM PYTHON</h2>
<img src="https://raw.githubusercontent.com/gorpo/PyQt5-Modern-Interface-/master/images/examples/browser.jpg" width="100%"></img>
 <h2 align="center">INTERPRETADOR PYTHON FEITO EM PYTHON</h2>
<img src="https://raw.githubusercontent.com/gorpo/PyQt5-Modern-Interface-/master/images/examples/python.jpg" width="100%"></img>
 <h2 align="center">SERVIDOR FLASK INTEGRADO</h2>
<img src="https://raw.githubusercontent.com/gorpo/PyQt5-Modern-Interface-/master/images/examples/flask.jpg" width="100%"></img>
 <h2 align="center">BANCO DE DADOS INTEGRADO FEITO COM SQLITE3</h2>
<img src="https://raw.githubusercontent.com/gorpo/PyQt5-Modern-Interface-/master/images/examples/bancodedados.jpg" width="100%"></img>


 <h2 align="center">OUTRAS FERRAMENTAS SERÃO LANÇADAS NA VERSAO 2.0</h2>


