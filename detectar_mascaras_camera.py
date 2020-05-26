
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# pegue as dimensões do quadro e construa um blob a partir dele
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	# passe o blob pela rede e obtenha as detecções de rosto
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# inicialize nossa lista de rostos, seus locais correspondentes e a lista de previsões da nossa rede de máscaras faciais
	faces = []
	locs = []
	preds = []
	# loop nas detecções
	for i in range(0, detections.shape[2]):
		# extrair a confiança (ou seja, probabilidade) associada à detecção
		confidence = detections[0, 0, i, 2]
		# filtrar detecções fracas, garantindo que a confiança seja maior que a confiança mínima
		if confidence > 0.5:
			# calcular as coordenadas (x, y) da caixa delimitadora para o objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# verifique se as caixas delimitadoras estão dentro das dimensões do quadro
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			# extrai o ROI da face, converte-o de pedido de canal BGR para RGB, redimensione-o para 224x224 e pré-processe-o
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
			# adicione as caixas de rosto e delimitadoras às respectivas listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# apenas faça previsões se pelo menos uma face for detectada
	if len(faces) > 0:
		# para uma inferência mais rápida, faremos previsões de lote em * todos * rostos ao mesmo tempo, em vez de previsões um por um no loop `for` acima
		preds = maskNet.predict(faces)
	# retorna duas tuplas dos locais de face e seus locais correspondentes
	return (locs, preds)




# carregar nosso modelo de detector de rosto serializado a partir do disco
print("[INFO] carregando modelo de detector de rosto ...")
prototxtPath = os.path.sep.join(["detectores/deploy.prototxt"])
weightsPath = os.path.sep.join(	["detectores/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# carregar o modelo do detector de máscara facial a partir do disco
print("[INFO] carregando modelo de detector de máscara facial ...")
maskNet = load_model("detectores/mask_detector.model")
# inicialize o fluxo de vídeo e permita que o sensor da câmera capte imagens
print("[INFO] iniciando video...")
#mude o valor da camera para 0 caso a camera seja built-in ou  troque pelo ip de sua camera-->> 'http://192.168.0.4:4747/mjpegfeed'
camera = 0
video = VideoStream(camera).start()
time.sleep(2.0)


# loop sobre o vídeo
while True:
	# pegue o quadro do fluxo de vídeo encadeado e redimensione-o para ter uma largura máxima de 400 pixels
	imagem = video.read()
	imagem = imutils.resize(imagem, width=400)
	# detectar rostos no quadro e determinar se eles estão usando uma máscara facial ou não
	(locs, preds) = detect_and_predict_mask(imagem, faceNet, maskNet)
	# circula sobre os locais de face detectados e seus locais correspondentes
	for (box, pred) in zip(locs, preds):
		# descompacte a caixa delimitadora e as previsões
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		# determine o rótulo e a cor da classe que usaremos para desenhar a caixa delimitadora e o texto
		label = "com mascara" if mask > withoutMask else "sem mascara"
		color = (0, 255, 0) if label == "com mascara" else (0, 0, 255)
		# inclui a probabilidade em uma label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# exibe os textos da label no frame
		cv2.putText(imagem, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(imagem, (startX, startY), (endX, endY), color, 2)

	# mostra o video
	cv2.imshow("Manicomio | Covid-19 Detector de Mascara.", imagem)
	# se a tecla "q" for pressionada quebra o loop e encerra camera
	tecla = cv2.waitKey(1) & 0xFF
	if tecla == ord("q"):
		break

# destroi as janelas
cv2.destroyAllWindows()
video.stop()