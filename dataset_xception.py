import os
import zipfile
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


"""
Script de creación de dataset desde Google Drive
"""

# Ruta de entrada y salida
dataset_dir = "/content/drive/My Drive/Dataset deepfakes"  # Carpeta con los ZIP
output_dir = "/content/processed_dataset"  # Carpeta para el dataset procesado
unzipped_dir = "/content/unzipped_dataset"  # Carpeta para los archivos descomprimidos

#Descomprimir los archivos ZIP
def extract_zips(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(input_dir, file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
    print("Archivos ZIP descomprimidos.")

# Reestructurar el dataset
def restructure_dataset(unzipped_dir, final_dir):
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

    # Procesar las imágenes reales (TEST10K/testB). Semilla de 42.
    real_dir = os.path.join(unzipped_dir, "TEST10K", "testB")
    real_images = find_images(real_dir)
    print(f"Encontradas {len(real_images)} imágenes reales en {real_dir}")
    train_real, temp_real = train_test_split(real_images, test_size=0.3, random_state=42)
    val_real, test_real = train_test_split(temp_real, test_size=0.5, random_state=42)

    for img in train_real:
        shutil.copy(img, os.path.join(train_dir, "real", os.path.basename(img)))
    for img in val_real:
        shutil.copy(img, os.path.join(val_dir, "real", os.path.basename(img)))
    for img in test_real:
        shutil.copy(img, os.path.join(test_dir, "real", os.path.basename(img)))

    # Procesar las imágenes falsas (resto de los ZIP)
    for folder in os.listdir(unzipped_dir):
        if folder != "TEST10K":  # Ignorar TEST10K, ya que contiene imágenes reales
            fake_dir = os.path.join(unzipped_dir, folder)
            fake_images = find_images(fake_dir)
            print(f"Encontradas {len(fake_images)} imágenes deepfake en {fake_dir}")
            train_fake, temp_fake = train_test_split(fake_images, test_size=0.3, random_state=42)
            val_fake, test_fake = train_test_split(temp_fake, test_size=0.5, random_state=42)

            for img in train_fake:
                shutil.copy(img, os.path.join(train_dir, "fake", os.path.basename(img)))
            for img in val_fake:
                shutil.copy(img, os.path.join(val_dir, "fake", os.path.basename(img)))
            for img in test_fake:
                shutil.copy(img, os.path.join(test_dir, "fake", os.path.basename(img)))

    print("Dataset reestructurado.")

# Configurar transformaciones*
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

# Crear loaders con batch de 16.
def create_loaders(data_dir, batch_size=16):
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=data_transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


extract_zips(dataset_dir, unzipped_dir)
restructure_dataset(unzipped_dir, output_dir)
train_loader, val_loader, test_loader = create_loaders(output_dir)
