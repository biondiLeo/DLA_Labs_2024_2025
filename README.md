# Deep Learning Applications 2024–2025
Repository contenente le soluzioni ai laboratori del corso **Deep Learning Applications 2024–2025**.

## 📋 Contenuti

| Lab   | Titolo                                 | Descrizione                                                                                                                                            | Notebook                                                       |
|--------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| Lab 1 | Reti MLP e CNN         | Implementazione di MLP e CNN su MNIST/CIFAR10, integrazione con Weights & Biases per il monitoraggio, tecniche di knowledge distillation con ResNet18 | [Lab1-CNNs.ipynb](Lab1-CNNs/Lab1-CNNs.ipynb)                   |
| Lab 3 | Analisi del Sentiment con Transformers | Classificazione binaria del sentiment con SVM e fine-tuning di DistilBERT, implementazione di LoRA per PEFT                                           | [Lab3-Transformers.ipynb](Lab3-Transformers/Lab3-Transformers.ipynb) |
| Lab 4 | Rilevamento OOD e Robustezza Avversariale | Tecniche di Out-of-Distribution detection (MSP, Autoencoder), attacchi FGSM, valutazione ROC/PR                                                       | [Lab4-OOD-2025.ipynb](Lab4-OOD/Lab4_OOD_2025.ipynb)            |

---

## 🚀 Setup Ambiente

### Opzione 1: Conda (Raccomandato)

**Per Lab 1 (CNNs):**

```bash
conda env create -f environment-dla.yml
conda activate DLA
```

**Per Lab 3 & 4 (Transformers & OOD):**

```bash
conda env create -f environment-transformers.yml
conda activate transformers
```

### Opzione 2: Pip

```bash
pip install -r requirements.txt
```

## 💻 Come Eseguire

Clona il repository:

```bash
git clone https://github.com/USERNAME/DLA_Labs_2024_2025.git
cd DLA_Labs_2024_2025
```

Configura l’ambiente (vedi sezione **Setup**)  
Naviga nel laboratorio desiderato:

```bash
cd Lab1-CNNs      # oppure Lab3-Transformers o Lab4-OOD
```

Avvia Jupyter:

```bash
jupyter lab
```

Apri il notebook corrispondente.

---

## ⚙️ Requisiti di Sistema

- **Python**: 3.13.x  
- **CUDA**: 12.7 (per supporto GPU)  
- **RAM**: Minimo 8GB raccomandato  
- **Storage**: ~2GB per il repository + spazio aggiuntivo per i dataset

---

## 📝 Note

I dataset utilizzati nei laboratori **non sono inclusi** nel repository per motivi di dimensione.  
Ogni laboratorio contiene istruzioni dettagliate su come scaricare e preparare i dati necessari.

- I modelli pre-addestrati di piccole dimensioni (<100MB) sono inclusi nel repository.
- Per dataset e modelli di grandi dimensioni, consultare i README specifici di ciascun laboratorio.
- Tutti i notebook sono stati testati con le versioni specificate in `requirements.txt`.

---

**Corso**: Deep Learning Applications  
**Autore**: Leonardo Biondi  
**Anno Accademico**: 2024–2025