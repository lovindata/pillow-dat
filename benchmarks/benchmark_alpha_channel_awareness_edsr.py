import cv2
from cv2 import dnn_superres
from PIL import Image

apple_image = cv2.imread(".github/apple.png")
model = dnn_superres.DnnSuperResImpl.create()
model.setModel("edsr", 2)
model.readModel("./dist/EDSR_x2.pb")  # Model weight needs to be downloaded
apple_image = model.upsample(apple_image)
Image.fromarray(cv2.cvtColor(apple_image, cv2.COLOR_BGR2RGB)).show()
