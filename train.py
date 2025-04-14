import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pretrainedmodels
import ssl

"""

Para la detección de deepfakes se utilizan modelos preentrenados de Xception y EfficientNet ajustándose algunos 
parámetros como la tasa de aprendizaje, la semilla, el batch, los folds y las épocas.

"""


# Deshabilita la verificación SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Configuración del dispositivo (La gpu que se va a usar para así hacerlo más rápido que con CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
print(f"Usando dispositivo: {device}")

# Transformaciones
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# Guardar checkpoints de los fold
def save_checkpoint(model, optimizer, epoch, fold, model_name):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, f"{model_name}_checkpoint_fold_{fold}_epoch_{epoch}.pth")
    print(f"Checkpoint guardado: {model_name}_checkpoint_fold_{fold}_epoch_{epoch}.pth")

# Graficar las pérdidas del modelo con un plot
def plot_loss(train_loss, val_loss, model_name):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Calcular métricas de análisis
def calculate_metrics(y_true, y_pred, y_probs, model_name, fold, results_file):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_probs)

    print(f"{model_name} Metrics (Fold {fold}):")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc_score:.4f}")

    # Guardar resultados en un archivo CSV
    with open(results_file, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([fold, model_name, tn, fp, fn, tp, f1, precision, recall, auc_score])

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.show()

#Entrenar y evaluar los modelos
def train_and_evaluate(model, optimizer, train_loader, val_loader, num_epochs, fold, model_name, results_file):
    criterion = nn.CrossEntropyLoss()
    train_loss_history, val_loss_history = [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        print(f"\n=== Fold {fold}, Epoch {epoch}/{num_epochs} ===")

        # Entrenamiento
        for i, (inputs, labels) in enumerate(train_loader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Entrenamiento - Procesando lote {i}/{len(train_loader)}", end="\r")
        train_loss = running_loss / len(train_loader)
        train_loss_history.append(train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader, start=1):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                print(f"Validación - Procesando lote {i}/{len(val_loader)}", end="\r")
        val_loss = val_loss / len(val_loader)
        val_loss_history.append(val_loss)

        print(f"\nFold {fold}, Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, fold, model_name)

    plot_loss(train_loss_history, val_loss_history, model_name)
    calculate_metrics(all_labels, all_preds, all_probs, model_name, fold, results_file)

if __name__ == "__main__":
    # Rutas del dataset
    data_dir = r"C:\Users\angie\PycharmProjects\generated_images\cleaned_dataset"
    batch_size = 8 #Procesar de 8 en 8
    num_folds = 5 #Número de folds
    results_file = "results_metrics.csv"

    # Crear archivo CSV y encabezado
    with open(results_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Fold", "Model", "TN", "FP", "FN", "TP", "F1-Score", "Precision", "Recall", "AUC"])

    # Cargar dataset completo
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    dataset_size = len(full_dataset)
    print(f"Total imágenes en dataset: {dataset_size}")

    # Validación cruzada de K-folds con semilla en 42
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kfold.split(full_dataset)):
        print(f"\n=== Iniciando Fold {fold + 1}/{num_folds} ===")

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # Modelo Xception
        xception_model = pretrainedmodels.__dict__['xception'](pretrained='imagenet')
        xception_model.last_linear = nn.Linear(xception_model.last_linear.in_features, 2)
        xception_model = xception_model.to(device)
        xception_optimizer = optim.Adam(xception_model.parameters(), lr=0.001) #Tasa de aprendizaje de 0.001

        print("Entrenando Xception...")
        #Se utilizan 10 épocas para el entrenamiento
        train_and_evaluate(xception_model, xception_optimizer, train_loader, val_loader, num_epochs=10,
                           fold=fold + 1, model_name="Xception", results_file=results_file)


## El modelo EfficientNet habría que descomentarlo en caso de querer entrenar este y no Xception, y comentar Xception.
"""      
    # Modelo EfficientNet
    efficientnet_model = EfficientNet.from_pretrained('efficientnet-b0')
    efficientnet_model._fc = nn.Linear(efficientnet_model._fc.in_features, 2)
    efficientnet_model = efficientnet_model.to(device)
    efficientnet_optimizer = optim.Adam(efficientnet_model.parameters(), lr=0.001)

    print("Entrenando EfficientNet...")
    train_and_evaluate(efficientnet_model, efficientnet_optimizer, train_loader, val_loader, num_epochs=10,
                       fold=fold + 1, model_name="EfficientNet", results_file=results_file)

"""



