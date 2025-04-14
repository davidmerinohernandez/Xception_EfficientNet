import os
import random
import uuid
from PIL import Image

# Configuración de carpetas de entrada y salida
input_dir = r"C:\Users\david\PycharmProjects\generated_images\processed_dataset"
output_dir = r"C:\Users\david\PycharmProjects\generated_images\cleaned_dataset"
image_size = (299, 299)  # Tamaño al que redimensionar las imágenes. En base al modelo

# Crear las carpetas de salida
for split in ['train', 'val', 'test']:
    for label in ['real', 'fake']:
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)



##Procesa y divide las imágenes en train, val y test.
##Cambia el nombre, redimensiona y equilibra las clasespata conseguir un dataset lo más homogéneo posible
def process_and_split_images(input_folder, output_folder, image_size, train_ratio=0.7, val_ratio=0.15):

    # Listar imágenes de cada clase
    real_images = [os.path.join(input_folder, 'train', 'real', img)
                   for img in os.listdir(os.path.join(input_folder, 'train', 'real'))
                   if img.endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff'))]

    fake_images = [os.path.join(input_folder, 'train', 'fake', img)
                   for img in os.listdir(os.path.join(input_folder, 'train', 'fake'))
                   if img.endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff'))]

    # Asegurar el mismo número de imágenes en ambas clases
    min_images = min(len(real_images), len(fake_images))
    real_images = random.sample(real_images, min_images)
    fake_images = random.sample(fake_images, min_images)

    print(f"Equilibrio de clases: {len(real_images)} imágenes en cada clase")

    # Combinar imágenes de ambas clases y dividir en train, val y test
    for label, images in [('real', real_images), ('fake', fake_images)]:
        random.shuffle(images)  # Barajar imágenes
        total_images = len(images)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        test_count = total_images - train_count - val_count

        splits = {
            'train': images[:train_count],
            'val': images[train_count:train_count + val_count],
            'test': images[train_count + val_count:]
        }

        for split, split_images in splits.items():
            target_path = os.path.join(output_folder, split, label)
            print(f"Procesando {len(split_images)} imágenes de {label} para {split}...")

            for img_path in split_images:
                new_name = f"{uuid.uuid4().hex}.jpg"  # Generar un nombre único para evitar sesgos
                new_path = os.path.join(target_path, new_name)

                # Redimensionar y guardar la imagen
                try:
                    with Image.open(img_path) as image:
                        image = image.convert("RGB")  # Asegurar tres canales
                        image = image.resize(image_size)
                        image.save(new_path, "JPEG")
                except Exception as e:
                    print(f"Error al procesar {img_path}: {e}")

    # Verificar las imágenes en las carpetas de salida
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(output_folder, split, label)
            count = len(os.listdir(path))
            print(f"Imágenes en {split}/{label}: {count}")

# Procesar imágenes y dividir en train, val, y test
process_and_split_images(input_dir, output_dir, image_size)

print("Preprocesamiento completo. Dataset limpio disponible en:", output_dir)
