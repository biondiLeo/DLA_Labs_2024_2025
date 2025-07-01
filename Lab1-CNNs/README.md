# Lab 1: Reti MLP e CNN

Implementazione di MLP e CNN su MNIST/CIFAR10, integrazione con Weights & Biases per il monitoraggio, tecniche di knowledge distillation utilizzando ResNet18.

## 📁 Contenuti

- `Lab1-CNNs.ipynb` - Notebook principale con tutti gli esercizi
- `best_model.pth` - Modello migliore salvato durante il training

## 🎯 Obiettivi del Laboratorio

1. **Implementazione MLP**: Creazione di Multi-Layer Perceptron con diverse configurazioni
2. **Residual Connections**: Studio dell'effetto delle skip connections in reti profonde
3. **CNN Architecture**: Implementazione di architetture convoluzionali di complessità crescente
4. **Knowledge Distillation**: Trasferimento di conoscenza da modelli teacher a student
5. **Monitoraggio**: Integrazione con Weights & Biases per tracking delle performance

## 🚀 Setup Ambiente

Attiva l'ambiente conda specifico per questo lab:

```bash
conda activate DLA
# oppure se non hai l'ambiente:
conda env create -f ../environment-dla.yml
conda activate DLA
```

## 📊 Risultati Sperimentali

### Exercise 1.1: MLP su MNIST

#### Baseline MLP vs Improved MLP

| Metrica         | Basic MLP | Improved MLP | Differenza |
|-----------------|-----------|--------------|------------|
| Test Accuracy   | 97.65%    | 98.12%       | +0.47%     |
| Test Loss       | 0.1519    | 0.0747       | -50.8%     |
| Train Accuracy  | 99.67%    | 98.76%       | -0.91%     |
| Val Accuracy    | 97.72%    | 98.18%       | +0.46%     |
| Train-Test Gap  | 2.02%     | 0.64%        | -68%       |

**📈 Weights & Biases Reports:**
- [Basic MLP](https://wandb.ai/leonardobiondi-universit-degli-studi-di-firenze/mnist-basic-mlp?nw=nwuserleonardobiondi)
- [Improved MLP](https://wandb.ai/leonardobiondi-universit-degli-studi-di-firenze/mnist-baseline-mlp?nw=nwuserleonardobiondi)

#### Analisi dell'Overfitting

**Basic MLP: Leggero Overfitting**
- Train accuracy: 99.67% vs Test: 97.65% (gap del 2.02%)
- Validation loss in crescita nelle epoche finali
- Il modello memorizza i dati di training senza generalizzare bene

**Enhanced MLP: Generalizzazione Ottima**
- Train accuracy: 98.76% vs Test: 98.12% (gap di soli 0.64%)
- Curve di learning più stabili
- Migliore capacità di generalizzazione su dati non visti

### Exercise 1.2: Residual Connections

#### Confronto Standard vs Residual MLP

| Profondità | Standard MLP Acc (%) | Residual MLP Acc (%) | Miglioramento (%) |
|------------|----------------------|----------------------|-------------------|
| 1          | 97.81                | 97.62                | -0.19             |
| 3          | 97.74                | 97.79                | +0.05             |
| 5          | 97.23                | 97.86                | +0.63             |
| 8          | 84.77                | 98.07                | **+13.30**        |

#### Evidenza del Vanishing Gradient Problem

**Depth 8 - Standard MLP:**
- Primo layer: gradiente norm = 0.261
- Ultimo layer: gradiente norm = 4.602 (**esplosione**)
- Layers intermedi: gradienti molto piccoli (~0.02–0.1)

**Depth 8 - Residual MLP:**
- Gradienti uniformi: tutti i layer mantengono norme tra 0.006–0.043
- Stabilità: nessuna esplosione o vanishing eccessivo
- Training efficace: validation accuracy del 98.07%

#### Analisi del Collasso nella Rete Profonda

Il crollo drammatico dell'MLP standard a depth 8 (84.77% vs 98.07%) dimostra:
- **Vanishing Gradients**: i layer iniziali ricevono gradienti troppo piccoli
- **Exploding Gradients**: i layer finali hanno gradienti eccessivi
- **Mancanza di Convergenza**: training accuracy solo 73.11% vs 99.46% nei residual

**Confronto training accuracy (depth 8):**
- Standard MLP: 73.11% (collasso del training)
- Residual MLP: 99.46% (convergenza ottimale)

### Exercise 1.3: CNN su CIFAR-10

#### Architetture a Confronto

| Modello     | Test Accuracy | Parametri | Architettura                      |
|-------------|---------------|-----------|-----------------------------------|
| SimpleCNN   | 74.85%        | 8.5M      | 2 conv layers (baseline)          |
| DeepCNN     | 79.35%        | 4.6M      | 4 conv layers, no residual        |
| ResNetCNN   | 84.94%        | 11.2M     | ResNet-18 with residual connections |

#### Progressione delle Performance

- SimpleCNN → DeepCNN: **+4.5%** di miglioramento
- DeepCNN → ResNetCNN: **+5.59%** di miglioramento
- **Miglioramento totale:** +10.1% (SimpleCNN → ResNetCNN)

### Exercise 2.2: Knowledge Distillation

#### Architetture Teacher-Student

**Teacher Model: ResNetCNN**
- Architettura: ResNet-18 modificata per CIFAR-10
- Parametri: 11,173,962 (~11.2M)
- Performance: **83.96%** test accuracy

**Student Model: SmallStudent**
- Architettura: CNN compatta con 3 layer convoluzionali + 2 fully connected
- Parametri: 2,159,114 (~2.2M) → riduzione dell'**80%** rispetto al teacher

#### Risultati Knowledge Distillation

| Modello           | Train Accuracy | Val Accuracy | Test Accuracy | Parametri | Miglioramento | Train-Test Gap |
|-------------------|----------------|--------------|---------------|-----------|---------------|----------------|
| Teacher (ResNet)  | 93.69%         | 85.92%       | 84.63%        | 11.2M     | —             | +9.06%         |
| Student Baseline  | 85.96%         | 75.88%       | 74.93%        | 2.2M      | baseline      | +11.03%        |
| Student + KD      | 86.65%         | 77.30%       | 76.82%        | 2.2M      | **+1.89%**    | +9.83%         |

**Hyperparameters Ottimali:**
- Temperatura: T=4.0
- Alpha: α=0.7
- Gap teacher-student: **7.23%**
- Riduzione parametri: **80.7%**

## 🔍 Conclusioni Chiave

1. **Efficacia del Dropout**: L'improved MLP dimostra come il dropout riduca significativamente l'overfitting (-68% nel train-test gap)

2. **Potenza delle Residual Connections**: Le skip connections risolvono completamente il vanishing gradient problem, permettendo training efficace di reti profonde

3. **Architetture Progressive**: Ogni incremento architetturale (SimpleCNN → DeepCNN → ResNetCNN) porta miglioramenti misurabili

4. **Knowledge Distillation Efficace**: Riduzione dell'80% dei parametri con solo 7.23% di gap nelle performance, dimostrando l'efficacia del trasferimento di conoscenza

## 💻 Esecuzione

1. Assicurati di avere l'ambiente `DLA` attivato
2. Apri il notebook: `jupyter lab Lab1-CNNs.ipynb`
3. Esegui le celle in sequenza
4. I risultati W&B saranno disponibili sui link forniti

## 📋 Dataset

- **MNIST**: Scaricato automaticamente tramite torchvision
- **CIFAR-10**: Scaricato automaticamente tramite torchvision
- **Preprocessing**: Normalizzazione e data augmentation implementati nel notebook

## ⚠️ Note Tecniche

- Richiesta GPU per training efficiente (CUDA 12.7)
- Tempo di esecuzione stimato: 2-3 ore per l'intero notebook
- RAM richiesta: minimo 8GB
- I modelli salvati sono disponibili in formato `.pth`

---

**Corso**: Deep Learning Applications \
**Autore**: Leonardo Biondi   
**Anno Accademico**: 2024-2025