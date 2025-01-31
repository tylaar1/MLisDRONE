import matplotlib.pyplot as plt
import os
import fitz 
from PIL import Image
import io


folder_path = r"C:\Users\user\Downloads\RL graphs\No sd"
pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])[:3]

images = []
for pdf in pdf_files:
    doc = fitz.open(os.path.join(folder_path, pdf))  
    pix = doc[0].get_pixmap() 
    img = Image.open(io.BytesIO(pix.tobytes("png"))) 
    images.append(img)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, img, title in zip(axes, images, pdf_files):
    ax.imshow(img)
    ax.set_title(os.path.splitext(title)[0])
    ax.axis("off")  
plt.tight_layout()
plt.show()
