# Product Requirements Document (PRD)
## Improved DeepFake Detection Using Whisper Features

### Overview
This codebase implements a deepfake audio detection system that leverages OpenAI's Whisper model features to improve detection accuracy. The system combines traditional audio frontend features (MFCC/LFCC) with Whisper's learned representations to create robust deepfake detection models.

### Core Architecture & Pipeline

#### 1. **Data Pipeline**
- **Input**: Audio files (16kHz sampling rate, 30-second chunks)
- **Preprocessing**:
  - Resampling to 16kHz
  - Stereo to mono conversion
  - Silence trimming (removes silence >0.2s, louder than 1% volume)
  - Padding/trimming to fixed length (480,000 samples = 30 seconds)
  - Normalization

#### 2. **Model Architecture Types**

**Base Models:**
- **SpecRNet**: Spectral Residual Network for audio classification
- **LCNN**: Light Convolutional Neural Network
- **MesoNet**: MesoInception4 architecture
- **RawNet3**: Raw waveform processing network

**Whisper-Enhanced Models:**
- **Whisper + SpecRNet**: Combines Whisper encoder features with SpecRNet classifier
- **Whisper + LCNN**: Combines Whisper encoder features with LCNN classifier  
- **Whisper + MesoNet**: Combines Whisper encoder features with MesoNet classifier

**Multi-Frontend Models:**
- **Whisper + Frontend + SpecRNet**: Combines Whisper features with traditional frontend (MFCC/LFCC) features
- **Whisper + Frontend + LCNN**: Similar combination with LCNN
- **Whisper + Frontend + MesoNet**: Similar combination with MesoNet

#### 3. **Feature Extraction Pipeline**

**Whisper Features:**
1. Audio → Log-Mel Spectrogram (80 mel bins, 16kHz, 25ms window, 10ms hop)
2. Log-Mel → Whisper Encoder (6-layer transformer with attention)
3. Whisper output → Feature reshaping for downstream models

**Traditional Frontend Features:**
- **MFCC**: 128 MFCC coefficients + delta + double-delta (384 total features)
- **LFCC**: 128 LFCC coefficients + delta + double-delta (384 total features)
- Features computed with 25ms window, 10ms hop, 512 FFT

#### 4. **Training Pipeline**

**Main Entry Points:**
- `train_and_test.py`: Complete training + evaluation pipeline
- `train_models.py`: Training only
- `evaluate_models.py`: Evaluation only

**Training Process:**
1. **Data Loading**: 
   - ASVspoof2021 DF dataset for training
   - In-The-Wild dataset for evaluation
   - Automatic oversampling to balance classes
2. **Model Initialization**:
   - Load Whisper encoder (tiny.en) with frozen/unfrozen parameters
   - Initialize downstream classifier (SpecRNet/LCNN/MesoNet)
3. **Training Loop**:
   - Binary classification (bonafide vs spoof)
   - BCEWithLogitsLoss
   - Adam optimizer with configurable learning rate
   - Optional cosine annealing scheduler (for RawNet3)
   - Best model selection based on validation accuracy
4. **Evaluation Metrics**:
   - Accuracy, Precision, Recall, F1-Score
   - AUC-ROC
   - Equal Error Rate (EER)

#### 5. **Configuration System**

**YAML Configuration Structure:**
```yaml
data:
  seed: 42

checkpoint:
  path: "path/to/checkpoint.pth"  # Empty for training, path for finetuning

model:
  name: "whisper_specrnet"  # Model type
  parameters:
    freeze_encoder: true    # Freeze Whisper parameters
    input_channels: 1       # Input channels
    frontend_algorithm: ["lfcc"]  # For multi-frontend models
  optimizer:
    lr: 0.0001
    weight_decay: 0.0001
```

#### 6. **Dataset Support**

**Training Datasets:**
- **ASVspoof2021 DF**: Primary training dataset
- **WaveFake**: Additional training data
- **FakeAVCeleb**: Additional training data
- **ASVspoof2019**: Additional training data

**Evaluation Datasets:**
- **In-The-Wild**: Real-world deepfake detection evaluation

**Dataset Features:**
- Automatic train/test/validation splits
- Speaker-aware splitting for evaluation
- Metadata tracking (attack types, speaker IDs)
- Flexible sampling (can limit number of samples)

#### 7. **Model Variants & Use Cases**

**Training from Scratch:**
- Use base models (specrnet, lcnn, mesonet, rawnet3)
- Or Whisper-enhanced models with `freeze_encoder: false`

**Finetuning:**
- Load pretrained weights via `checkpoint.path`
- Set `freeze_encoder: true` to freeze Whisper parameters
- Lower learning rates recommended

**Multi-Frontend Fusion:**
- Combines Whisper features with traditional audio features
- Uses `whisper_frontend_*` model variants
- Requires `input_channels: 2` for concatenated features

#### 8. **Key Technical Details**

**Whisper Integration:**
- Uses Whisper tiny.en encoder (6 layers, 384 hidden size)
- Pre-computed mel spectrograms (80 bins, 3000 frames)
- Positional embeddings and multi-head attention
- Feature extraction at 1500 frames (after conv layers)

**Audio Processing:**
- Fixed 30-second audio chunks
- 16kHz sampling rate
- Automatic silence removal and padding
- Batch processing with configurable batch sizes

**Training Infrastructure:**
- PyTorch-based training loop
- GPU/CPU automatic detection
- Reproducible experiments with seed setting
- Model checkpointing and best model selection

#### 9. **Usage Examples**

**Full Training Pipeline:**
```bash
python train_and_test.py \
  --asv_path ../datasets/ASVspoof2021/DF \
  --in_the_wild_path ../datasets/release_in_the_wild \
  --config configs/training/whisper_specrnet.yaml \
  --batch_size 8 --epochs 10 \
  --train_amount 100000 --valid_amount 25000
```

**Finetuning:**
```bash
python train_and_test.py \
  --config configs/finetuning/whisper_specrnet.yaml \
  --epochs 5  # Lower epochs for finetuning
```

#### 10. **Performance & Evaluation**

**Metrics Computed:**
- Classification accuracy
- Precision, Recall, F1-Score
- AUC-ROC for threshold-independent evaluation
- Equal Error Rate (EER) for biometric-style evaluation

**Model Selection:**
- Best model based on validation accuracy
- Automatic checkpointing of best performing model
- Configurable evaluation on held-out test sets

### Dependencies & Setup

**Core Dependencies:**
- Python 3.8
- PyTorch 1.11.0
- TorchAudio 0.11
- OpenAI Whisper (specific commit)
- Librosa 0.9.2
- Asteroid-filterbanks 0.4.0

**Setup Process:**
1. Run `bash install.sh` for dependency installation
2. Run `python download_whisper.py` to download Whisper encoder
3. Download required datasets (ASVspoof2021, In-The-Wild)
4. Configure YAML files for desired model variants

### Key Innovations

1. **Whisper Feature Integration**: First use of Whisper's learned audio representations for deepfake detection
2. **Multi-Frontend Fusion**: Combines modern transformer features with traditional audio features
3. **Flexible Architecture**: Supports multiple downstream classifiers with Whisper features
4. **Comprehensive Evaluation**: Multiple datasets and metrics for robust evaluation
5. **Reproducible Research**: Complete configuration system and seed management

This system represents a significant advancement in deepfake audio detection by leveraging state-of-the-art speech recognition models for improved feature extraction and classification performance.
