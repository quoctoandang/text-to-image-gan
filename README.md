# 🎨 Text-to-Image GAN

> Generate realistic images from text descriptions using Generative Adversarial Networks

## 📋 Overview

This project implements multiple **Text-to-Image Generation** models using different GAN architectures. The models are trained on the **COCO 2017 dataset** to generate realistic images from natural language descriptions.

### 🤖 Implemented Models

1. **DCGAN (Deep Convolutional GAN)** - `text-to-image-dcgan.ipynb`
2. **WGAN (Wasserstein GAN)** - `text-to-image-wgan.ipynb`
3. **GALIP (Generative Adversarial CLIP)** - `text_to_image_galip.ipynb`

## ✨ Key Features

- 🖼️ **Image Generation from Text** - Convert text descriptions into realistic images
- 📊 **Multiple GAN Architectures** - Compare different approaches (DCGAN, WGAN, GALIP)
- 🎯 **COCO Dataset Integration** - Trained on MS COCO 2017 with 118K+ training images
- 📈 **Model Evaluation** - FID (Fréchet Inception Distance) and IS (Inception Score) metrics
- 🔄 **Text Embeddings** - Using SBERT and CLIP for semantic understanding
- 📉 **Training Visualization** - Weights & Biases (wandb) integration

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning Framework** | PyTorch |
| **Text Encoding** | Sentence-BERT, CLIP (OpenAI) |
| **Dataset** | MS COCO 2017 |
| **Evaluation Metrics** | FID, Inception Score |
| **Training Tracking** | Weights & Biases (wandb) |
| **Image Processing** | PIL, torchvision |

## 📁 Project Structure

```
text-to-image-gan/
├── text-to-image-dcgan.ipynb    # DCGAN implementation
├── text-to-image-wgan.ipynb     # WGAN implementation
├── text_to_image_galip.ipynb    # GALIP implementation
└── README.md                     # Project documentation
```

## 🗃️ Dataset

### MS COCO 2017 Dataset
- **Training Images:** 118,287 images
- **Validation Images:** 5,000 images
- **Test Images:** 40,670 images
- **Captions:** ~5 captions per image
- **Average Caption Length:** ~10-12 words

### Data Structure
```
coco2017/
├── train2017/           # Training images
├── val2017/             # Validation images
├── test2017/            # Test images
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

## 🧠 Model Architectures

### 1. DCGAN (Deep Convolutional GAN)
- **Generator:** Upsampling with transposed convolutions
- **Discriminator:** Convolutional layers with batch normalization
- **Text Encoding:** SBERT (384-dimensional embeddings)
- **Image Size:** 256x256 pixels

### 2. WGAN (Wasserstein GAN)
- **Loss Function:** Wasserstein distance
- **Gradient Penalty:** For stable training
- **Spectral Normalization:** In discriminator layers
- **Text Encoding:** SBERT

### 3. GALIP (Generative Adversarial CLIP)
- **Text Encoder:** CLIP ViT-B/32
- **Image Encoder:** CLIP vision encoder
- **Joint Embedding Space:** CLIP latent space
- **Advanced Architecture:** State-of-the-art results

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd text-to-image-gan
```

2. **Install dependencies**
```bash
pip install torch torchvision
pip install sentence-transformers
pip install torchmetrics
pip install torch-fidelity
pip install wandb
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

3. **Download COCO 2017 Dataset**
```bash
# Download from http://cocodataset.org/#download
# Or use Kaggle dataset: https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
```

### Training

1. **Configure Wandb (Optional)**
```python
import wandb
wandb.login(key="your_api_key")
wandb.init(project="text-to-image-gan")
```

2. **Run Training Notebook**
- Open your preferred notebook (DCGAN/WGAN/GALIP)
- Update dataset paths in the notebook
- Execute cells sequentially
- Monitor training on Wandb dashboard

### Usage Example

```python
# Generate image from text
caption = "a black cat sitting on a wooden table"

# Encode text
text_embedding = encode_captions([caption])

# Generate image
with torch.no_grad():
    fake_image = generator(noise, text_embedding)
    
# Display result
plt.imshow(fake_image.permute(1, 2, 0).cpu())
plt.title(caption)
plt.show()
```

## 📊 Evaluation Metrics

### Fréchet Inception Distance (FID)
Measures the distance between real and generated image distributions
- **Lower is better**
- Range: 0 to ∞

### Inception Score (IS)
Evaluates quality and diversity of generated images
- **Higher is better**
- Range: 1 to ∞

## 🎯 Training Configuration

| Parameter | DCGAN | WGAN | GALIP |
|-----------|-------|------|-------|
| **Image Size** | 256x256 | 256x256 | 256x256 |
| **Batch Size** | 32-64 | 32-64 | 16-32 |
| **Learning Rate** | 0.0002 | 0.0001 | 0.0001 |
| **Text Embedding** | SBERT-384 | SBERT-384 | CLIP-512 |
| **Training Epochs** | 50-100 | 50-100 | 50-100 |

## 📈 Results

### Sample Generations

Input: `"a dog playing in the park"`
- Generated realistic images of dogs in outdoor settings

Input: `"a red sports car on the highway"`
- Produced clear images of red vehicles on roads

Input: `"a sunset over mountains"`
- Created scenic landscape images

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller subset of data
   - Enable gradient checkpointing

2. **Training Instability**
   - Adjust learning rates
   - Use gradient clipping
   - Try different GAN architectures (WGAN recommended)

3. **Poor Image Quality**
   - Train for more epochs
   - Increase model capacity
   - Use better text embeddings (CLIP)

## 📚 References

- [MS COCO Dataset](http://cocodataset.org/)
- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [WGAN Paper](https://arxiv.org/abs/1701.07875)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084)

## 📝 License

This project is developed for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

---
⭐ If you find this project useful, please give it a star!
