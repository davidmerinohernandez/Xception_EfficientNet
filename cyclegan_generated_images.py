import os
import shutil

# Directorio de origen y destino
source_dir = r"C:\Users\david\PycharmProjects\generated_images\images_pix2pix"
destination_dir = os.path.join(source_dir, "pix2pix_generated_images")

# Crear carpeta de destino si no existe
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Filtrar y copiar imágenes
for file_name in os.listdir(source_dir):
    if "fake_B" in file_name:  # Buscar imágenes con 'fake_A' en su nombre
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.copy(source_path, destination_path)

print(f"Las imágenes con 'fake_B' en su título se han copiado a {destination_dir}")
