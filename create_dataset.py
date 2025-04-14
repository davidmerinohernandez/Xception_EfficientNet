import os
import zipfile
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


"""
Script de creación de un dataset con entrenamiento, validación y test
"""
# Rutas
dataset_dir = r"C:\Users\david\PycharmProjects\generated_images\Dataset deepfakes"
output_dir = r"C:\Users\david\PycharmProjects\generated_images\processed_dataset"
unzipped_dir = r"C:\Users\david\PycharmProjects\generated_images\unzipped_dataset"

#Descomprimir los archivos ZIP
def extract_zips(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(input_dir, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
    print("Archivos ZIP descomprimidos.")

#  Reestructurar el dataset
def restructure_dataset(unzipped_dir, final_dir, real_dir):
    train_dir = os.path.join(final_dir, "train")
    val_dir = os.path.join(final_dir, "val")
    test_dir = os.path.join(final_dir, "test")

    # Crear carpetas para real y fake
    for split in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(split, "real"), exist_ok=True)
        os.makedirs(os.path.join(split, "fake"), exist_ok=True)

    # Función para buscar imágenes recursivamente
    def find_images(directory):
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', '.webp', '.tiff')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    # Distribuir imágenes en train, val y test
    def distribute_images(images, output_class_dir, train_dir, val_dir, test_dir):
        total_images = len(images)
        train_count = int(total_images * 0.7)
        val_count = int(total_images * 0.15)
        test_count = total_images - train_count - val_count

        train_images, temp_images = train_test_split(images, train_size=train_count, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=test_count, random_state=42)

        for img in train_images:
            shutil.copy(img, os.path.join(train_dir, output_class_dir, os.path.basename(img)))
        for img in val_images:
            shutil.copy(img, os.path.join(val_dir, output_class_dir, os.path.basename(img)))
        for img in test_images:
            shutil.copy(img, os.path.join(test_dir, output_class_dir, os.path.basename(img)))

    # Procesar imágenes reales
    real_images = find_images(real_dir)
    print(f"Encontradas {len(real_images)} imágenes reales en {real_dir}")
    distribute_images(real_images, "real", train_dir, val_dir, test_dir)

    # Procesar las imágenes falsas (extraídas de los ZIP)
    for folder in os.listdir(unzipped_dir):
        fake_dir = os.path.join(unzipped_dir, folder)
        if os.path.isdir(fake_dir):  # Asegurarse de que es un directorio
            fake_images = find_images(fake_dir)
            print(f"Encontradas {len(fake_images)} imágenes deepfake en {fake_dir}")
            distribute_images(fake_images, "fake", train_dir, val_dir, test_dir)

    print("Dataset reestructurado.")

#Configurar transformaciones
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Crear loaders con batch de 16 imágenes. Val, train y test.
def create_loaders(data_dir, batch_size=16):
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader

real_images_dir = os.path.join(dataset_dir, "testA")  # Carpeta con imágenes reales
extract_zips(dataset_dir, unzipped_dir)
restructure_dataset(unzipped_dir, output_dir, real_images_dir)
train_loader, val_loader, test_loader = create_loaders(output_dir)
