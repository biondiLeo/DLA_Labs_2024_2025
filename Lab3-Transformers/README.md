# Lab 3: Analisi del Sentiment con Transformers

Classificazione binaria del sentiment utilizzando SVM tradizionali e fine-tuning di DistilBERT, implementazione di LoRA per Parameter-Efficient Fine-Tuning + Implementazione extra finale di KD-LoRA seguendo i principi del paper "Combining Knowledge Distillation with Parameter-Efficient Fine-Tuning" - *Azimi et al., 2024 - 4th NeurIPS Efficient Natural Language and Speech Processing Workshop*  
Disponibile su: https://arxiv.org/pdf/2410.20777

## 📁 Contenuti

- `Lab3-Transformers.ipynb` - Notebook principale con tutti gli esercizi di sentiment analysis

## 🎯 Obiettivi del Laboratorio

1. **Dataset Exploration**: Analisi approfondita del Cornell Rotten Tomatoes dataset
2. **Feature Extraction**: Utilizzo di DistilBERT come estrattore di features per classificazione tradizionale
3. **Fine-tuning End-to-End**: Adattamento completo del modello transformer al task specifico
4. **Parameter-Efficient Fine-Tuning**: Implementazione di LoRA per ottimizzazione delle risorse

## 🚀 Setup Ambiente

Attiva l'ambiente conda specifico per questo lab:

```bash
conda activate transformers
# oppure se non hai l'ambiente:
conda env create -f ../environment-transformers.yml
conda activate transformers
```

**Dipendenza aggiuntiva richiesta:**
```bash
pip install peft
```

## 📊 Risultati Sperimentali

### Exercise 1: Sentiment Analysis Warm-up

#### 1.1 Dataset Exploration - Cornell Rotten Tomatoes

**Caratteristiche del Dataset:**
- **Train**: 8,530 esempi
- **Validation**: 1,066 esempi  
- **Test**: 1,066 esempi
- **Bilanciamento perfetto**: 50% negativi, 50% positivi
- **Encoding**: `0 = neg`, `1 = pos`

**Qualità dei Dati:**
- Recensioni di lunghezza variabile
- Linguaggio naturale e descrittivo
- Diversi generi cinematografici
- Perfettamente bilanciato per evitare bias

#### 1.2 DistilBERT Pre-trained Model

**Architettura DistilBERT:**
- **Parametri**: 66M parametri
- **Layers**: 6 transformer layers
- **Vocabulary**: 30k token
- **Hidden Size**: 768 dimensioni
- **Tokenizzazione intelligente**: sub-token per parole complesse

**Gestione del Testo:**
- Token speciali [CLS] e [SEP] aggiunti automaticamente
- Token [CLS] cattura il significato globale della frase
- Adattamento dinamico a frasi di lunghezza diversa

#### 1.3 Stable Baseline (DistilBERT + SVM)

**Risultati Baseline:**
- **Accuratezza Validation**: **81.43%**
- **Accuratezza Test**: **79.46%**
- **Performance bilanciate**: precision e recall simili per entrambe le classi

**Vantaggi dell'Approccio:**
- Significativamente più veloce e leggero
- Estrazione features una tantum
- Risorse computazionali contenute
- Performance competitive con modello frozen

### Exercise 2: Fine-tuning DistilBERT

#### 2.1 Token Preprocessing

Implementazione di pipeline di tokenizzazione efficiente utilizzando `Dataset.map` di HuggingFace per conversione lazy da stringhe a token IDs.

#### 2.2 Model Architecture Setup

**DistilBertForSequenceClassification:**
```
Parametri trainable: 66,955,010
Parametri totali: 66,955,010
Percentuale trainable: 100.0%

Classification head:
- Input features: 768
- Output features: 2
- Parametri aggiuntivi: 1,538
```

**Componenti Architettura:**
- 6 TransformerBlock con DistilBertSdpaAttention
- Pre-classifier: Linear(768→768)
- Classifier: Linear(768→2)
- Dropout: 0.2

#### 2.3 Fine-tuning Results

**Training Progress:**

| Epoca | Loss Training | Loss Validation | Accuratezza Validation |
|-------|---------------|-----------------|------------------------|
| 1     | 0.398         | 0.371           | 84.24%                 |
| 2     | 0.269         | 0.414           | 83.77%                 |
| 3     | 0.091         | 0.617           | 84.15%                 |

**Performance Finali:**
- **Accuratezza Test**: **83.58%**
- **Loss Test**: **0.3988**
- **F1-Score**: **0.8358**
- **Miglioramento vs Baseline**: **+4.12%** (79.46% → 83.58%)

**Osservazioni:**
- Evidenti segni di overfitting dalla seconda epoca
- 2 epoche potrebbero rappresentare il compromesso ottimale
- Performance bilanciate su entrambe le classi

### Exercise 3: Parameter-Efficient Fine-tuning (LoRA)

#### 3.1 LoRA Implementation

**Training Progress:**

| Epoca | Loss Training | Loss Validation | Accuratezza Validation |
|-------|---------------|-----------------|------------------------|
| 1     | 0.416         | 0.440           | 82.36%                 |
| 2     | 0.331         | 0.363           | 84.05%                 |
| 3     | 0.274         | 0.373           | 84.15%                 |

**Performance Finali:**
- **Accuratezza Test**: **82.93%**
- **Loss Test**: **0.4319**
- **Tempo Training**: **1:28 minuti**

**Efficienza LoRA:**
- **Parametri Trainable**: **887,042** su **67,842,052** totali (**1.31%**)
- **Riduzione Parametri**: **76.5×** meno parametri da addestrare
- **Risparmio Memoria**: **98.7%**

#### EXTRA: KD-LoRA Implementation
**Teacher Model Setup:**
- **Architettura**: BERT-base (110M parametri)
- **Training**: 2 epoche specifiche sul dataset
- **Performance Teacher**: **87.15%** validation accuracy
- **Ruolo**: Fornisce soft targets per la distillazione

**Training Progress KD-LoRA:**

| Epoca | Loss Combinata | Loss Task | Loss Distillation | Accuratezza Validation |
|-------|----------------|-----------|-------------------|------------------------|
| 1     | 0.428          | 0.445     | 0.411             | 82.84%                 |
| 2     | 0.342          | 0.359     | 0.325             | 84.24%                 |
| 3     | 0.281          | 0.291     | 0.271             | 84.43%                 |

**Performance Finali:**
- **Accuratezza Test**: **83.11%**
- **Parametri Trainable**: **887,042** (identici a LoRA)
- **Temperature**: **3.0** per soft target smoothing
- **Alpha**: **0.5** (bilanciamento task/distillation loss)

**Miglioramenti vs LoRA:**
- **Performance**: 83.11% vs 82.93% LoRA (+0.18 punti percentuali)
- **Knowledge Transfer Rate**: 6.5% del gap teacher-student recuperato
- **Performance Retention**: 100.2% delle performance di LoRA
- **Zero Overhead**: Stessa velocità di inferenza del modello LoRA

## 📈 Confronto Completo degli Approcci

| Approccio            | Tipo Addestramento              | Test Accuracy | Parametri Trainable | Tempo Training | Performance Gap |
|----------------------|----------------------------------|---------------|---------------------|----------------|-----------------|
| **Baseline (SVM)**   | Feature extraction + classifier | 79.46%        | 0 (frozen)          | ~40 secondi    | baseline        |
| **Full Fine-tuning** | End-to-end completo             | **83.58%**    | 67.8M (100%)        | ~3 minuti      | **+4.12%**      |
| **LoRA Fine-tuning** | Parameter-efficient adapter     | **82.93%**    | **0.9M (1.31%)**    | **1.5 minuti** | **+3.47%**      |
| **KD-LoRA**          | Parameter-efficient + distillation | **83.11%**    | **0.9M (1.31%)**    | **1.6 minuti** | **+3.65%**      |

## 🔍 Conclusioni Chiave

### 1. Efficacia del Fine-tuning
Il fine-tuning end-to-end dimostra chiari vantaggi rispetto al feature extraction:
- **+4.12%** di miglioramento in accuratezza
- Adattamento di tutti i 67M parametri al task specifico
- Rappresentazioni più efficaci per sentiment analysis

### 2. Eccellenza di LoRA
LoRA raggiunge il **99.2%** delle performance del full fine-tuning con solo l'**1.31%** dei parametri:
- **Trade-off ideale** performance/efficienza
- **98.7% risparmio memoria**
- **2× più veloce** nel training
- Perdita trascurabile: **-0.65%** vs full fine-tuning

*Considerazioni extra su KD-LoRA*:
- KD-LoRA raggiunge il **99.4%** delle performance del Full Fine-Tuning con solo l'**1.3%** dei parametri
- Riduzione del **37%** del gap di performance rispetto al solo LoRA
- **Validazione empirica** dell'efficacia del knowledge transfer in contesti parameter-efficient

### 3. Robustezza delle Soluzioni
- Tutte le implementazioni mantengono performance bilanciate
- Nessun bias significativo verso una classe specifica
- LoRA dimostra stabilità e praticità per applicazioni production

### 4. Impatto Computazionale
- **SVM**: Veloce ma limitato in performance
- **Full Fine-tuning**: Ottimale ma costoso
- **LoRA**: Compromesso perfetto per applicazioni reali

## 💻 Esecuzione

1. Assicurati di avere l'ambiente `transformers` attivato
2. Installa PEFT: `pip install peft`
3. Apri il notebook: `jupyter lab Lab3-Transformers.ipynb`
4. Esegui le celle in sequenza
5. Il dataset sarà scaricato automaticamente da HuggingFace

## 📋 Dataset

- **Cornell Rotten Tomatoes**: Scaricato automaticamente tramite HuggingFace Datasets
- **Preprocessing**: Tokenizzazione automatica con DistilBERT tokenizer
- **Splits**: Train/Validation/Test predefiniti

## ⚠️ Note Tecniche

- **GPU Raccomandabile**: Per training efficiente (LoRA funziona anche su CPU)
- **RAM**: Minimo 8GB, raccomandati 16GB per full fine-tuning
- **Versioni**: Compatibilità testata con transformers>=4.51.1 e peft>=0.15.2

Se incontri problemi di compatibilità PEFT/Transformers, il notebook include un **Custom Trainer** che risolve automaticamente i conflitti di versione tra le librerie.

---

**Corso**: Deep Learning Applications \
**Autore**: Leonardo Biondi   
**Anno Accademico**: 2024-2025