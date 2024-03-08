import cv2
from cv2 import dnn_superres
from PIL import Image

lumine_image = cv2.imread(".github/lumine.png")
model = dnn_superres.DnnSuperResImpl.create()
model.setModel("edsr", 2)
model.readModel("./dist/EDSR_x2.pb")  # Model weight needs to be download
lumine_image = model.upsample(lumine_image)
Image.fromarray(cv2.cvtColor(lumine_image, cv2.COLOR_BGR2RGB)).show()
