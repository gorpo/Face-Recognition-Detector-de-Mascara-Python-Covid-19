
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


# inicialize a taxa de aprendizado inicial, número de épocas para treinamento e tamanho do lote
inicia_apredizado = 1e-4
quantidade_treinos = 30
tamanho_lote = 32 #batch_size
# pegue a lista de imagens em nosso diretório de conjunto de dados e inicialize a lista de dados (ou seja, imagens) e imagens de classe
print("[INFO] carregando imagens...")
caminho_imagens = list(paths.list_images("dataset"))
dados = []
labels = []

# loop sobre os caminhos da imagem
for imagePath in caminho_imagens:
	# extrai o rótulo da classe do nome do arquivo
	label = imagePath.split(os.path.sep)[-2]
	# carrega a imagem de entrada (224x224) e pré-processa
	imagem = load_img(imagePath, target_size=(224, 224))
	imagem = img_to_array(imagem)
	imagem = preprocess_input(imagem)

	# atualiza as listas de dados e etiquetas, respectivamente
	dados.append(imagem)
	labels.append(label)

# converte os dados e rótulos em matrizes NumPy
dados = np.array(dados, dtype="float32")
labels = np.array(labels)

# executar codificação one-hot nas etiquetas
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# particione os dados em divisões de treinamento e teste usando 75% dos dados para treinamento e os 25% restantes para teste
(trainX, testX, trainY, testY) = train_test_split(dados, labels, test_size=0.20, stratify=labels, random_state=42)

# construir o gerador de imagens de treinamento para aumento de dados
gerador = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# carregar a rede MobileNetV2, garantindo que os conjuntos de camadas FC principais sejam deixados de lado
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construa a cabeça do modelo que será colocado em cima do modelo base
modelo_cabeca = baseModel.output
modelo_cabeca = AveragePooling2D(pool_size=(7, 7))(modelo_cabeca)
modelo_cabeca = Flatten(name="flatten")(modelo_cabeca)
modelo_cabeca = Dense(128, activation="relu")(modelo_cabeca)
modelo_cabeca = Dropout(0.5)(modelo_cabeca)
modelo_cabeca = Dense(2, activation="softmax")(modelo_cabeca)

# coloque o modelo -FACE RECOGNITION- principal sobre o modelo base (este se tornará o modelo real que iremos treinar)
modelo = Model(inputs=baseModel.input, outputs=modelo_cabeca)

# percorre todas as camadas no modelo base e as congela para que elas * não * sejam atualizadas durante o primeiro processo de treinamento
for layer in baseModel.layers:
	layer.trainable = False

# compile nosso modelo
print("[INFO] compilando o modelo...")
otimizador = Adam(lr=inicia_apredizado, decay=inicia_apredizado / quantidade_treinos)
modelo.compile(loss="binary_crossentropy", optimizer=otimizador, metrics=["accuracy"])

# treinar a cabeça
print("[INFO] treinando para reconhecer a  cabeça...")
cabeca = modelo.fit(gerador.flow(trainX, trainY, batch_size=tamanho_lote), steps_per_epoch=len(trainX) // tamanho_lote, validation_data=(testX, testY), validation_steps=len(testX) // tamanho_lote, epochs=quantidade_treinos)

# faça previsões(predicções) no conjunto de testes
print("[INFO] avaliação de rede neural...")
predIdxs = modelo.predict(testX, batch_size=tamanho_lote)

# para cada imagem no conjunto de testes, precisamos encontrar o índice do rótulo com a maior probabilidade prevista correspondente
predIdxs = np.argmax(predIdxs, axis=1)

# mostrar um relatório de classificação
print(classification_report(testY.argmax(axis=1), predIdxs,	target_names=lb.classes_))

# salve o modelo
print("[INFO] salvando o modelo detector de mascara...")
modelo.save("detectores/mask_detector.model", save_format="h5")

# traçar a perda e a precisão do treinamento
N = quantidade_treinos
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), cabeca.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), cabeca.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), cabeca.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), cabeca.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("face_detector/plot.png")
