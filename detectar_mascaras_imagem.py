
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


# carregar nosso modelo de detector de rosto serializado a partir do disco
print("[INFO] carregando modelo detector de faces...")
prototxtPath = os.path.sep.join(["detectores/deploy.prototxt"])
weightsPath = os.path.sep.join(	["detectores/res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)
# carrega o detector de mascara facialload the face mask detector model from disk
print("[INFO] carregando detector de mascara facial...")
modelo = load_model("detectores/mask_detector.model")
# carregue a imagem, clone-a e pegue as dimensões espaciais da imagem
imagem = cv2.imread('exemplos/1.png')
origem = imagem.copy()
(h, w) = imagem.shape[:2]
# construir um blob a partir da imagem
blob = cv2.dnn.blobFromImage(imagem, 1.0, (300, 300), (104.0, 177.0, 123.0))
# passe o blob pela rede e obtenha as detecções de rosto
print("[INFO] computando deteccoes faciais...")
net.setInput(blob)
deteccoes = net.forward()


# loop sobre as detecçoes
for i in range(0, deteccoes.shape[2]):
	# extrair a confiança (ou seja, probabilidade) associada à detecção
	confianca = deteccoes[0, 0, i, 2]
	# filtrar detecções fracas, garantindo que a confiança seja maior que a confiança mínima
	if confianca > 0.5:
		# calcular as coordenadas (x, y) da caixa delimitadora para o objeto
		box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		# verifique se as caixas delimitadoras estão dentro das dimensões do quadro
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
		# extrai o ROI da face, converte-o de pedido de canal BGR para RGB, redimensione-o para 224x224 e pré-processe
		face = imagem[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
		# passe o rosto pelo modelo para determinar se o rosto tem uma máscara ou não
		(mask, withoutMask) = modelo.predict(face)[0]
		# determine a label e a cor da classe que usaremos para desenhar a caixa delimitadora e o texto
		label = "com mascara" if mask > withoutMask else "sem mascara"
		color = (0, 255, 0) if label == "com mascara" else (0, 0, 255)
		# inclua a probabilidade na label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# exibir a label e o retângulo da caixa delimitadora no quadro de saída
		cv2.putText(imagem, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(imagem, (startX, startY), (endX, endY), color, 2)

# Mostra a imagem final
cv2.imshow("Manicomio | Covid-19 Detector de Mascara.", imagem)
cv2.waitKey(0)