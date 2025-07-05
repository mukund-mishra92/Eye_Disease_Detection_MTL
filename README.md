# 👁️ Eye Disease Detection using Multi-Task Learning (MTL)

A unified deep learning system to **classify diabetic retinopathy severity** and **segment lesions** from retinal images. This repo includes model training, evaluation, inference, visualization, and Streamlit-based UI.

---

## 🚀 Quick Commands

### 🖼 Launch the Web App
```bash
streamlit run app.py
```

### 🔍 Inference on Single Image

#### 1. Classification Only:
```bash
python -m modules.MTL_with_Router.inference_cls
```

#### 2. Segmentation Only:
```bash
python -m modules.MTL_with_Router.inference_seg
```

---

### 🏋️ Train the Model

#### A. Basic Multitask Training
```bash
python -m modules.MTL_with_Router.train
```

#### B. With Dynamic Router Logic
```bash
python -m modules.MTL_with_Router.train1
```

#### C. With Optimized Parameters (from `config.json`)
```bash
python -m modules.MTL_with_Router.train_2
```

---

### 📈 Visualize Metrics
```bash
python -m modules.MTL_with_Router.plot_logs
```

---

### 🧪 Evaluate Final Model
```bash
python -m modules.MTL_with_Router.evaluate
```

---

## ⚙️ Configuration (`config.json`)

```json
{
  "batch_size": 8,
  "epochs": 30,
  "lr": 0.0001,
  "weight_decay": 0.0005,
  "cls_loss_weight": 1.0,
  "seg_loss_weight": 1.0,
  "train_cls_image_dir": "Preprocessed_Data/classification/train/train_images",
  "train_cls_label_json": "Preprocessed_Data/classification/train/train_iamge_lable_mapping.json",
  "val_cls_image_dir": "Preprocessed_Data/classification/test/test_images",
  "val_cls_label_json": "Preprocessed_Data/classification/test/test_image_label_mapping.json",
  "train_seg_image_dir": "Preprocessed_Data/segmentation/train/train_images",
  "train_seg_mask_dir": "Preprocessed_Data/segmentation/train/groundtruth",
  "val_seg_image_dir": "Preprocessed_Data/segmentation/test/test_images",
  "val_seg_mask_dir": "Preprocessed_Data/segmentation/test/groundtruth"
}
```

---

## 📁 Folder Structure

```
├── app.py
├── config.json
├── models/                 # Saved models
├── logs/                   # Training logs and plots
├── modules/
│   └── MTL_with_Router/
│       ├── model.py        # MTL model (shared encoder + task heads)
│       ├── datast.py       # MultiTask dataset class
│       ├── train.py        # Simple multitask training
│       ├── train1.py       # With dynamic router logic
│       ├── train_2.py      # Uses config for optimization
│       ├── inference_cls.py
│       ├── inference_seg.py
│       ├── plot_logs.py
│       ├── evaluate.py
│       ├── losses.py       # Focal loss, Dice loss etc.
├── requirements.txt
```

---

## 🧠 Features

- ✅ Multitask model with shared ResNet encoder
- 🧩 Dual-head architecture: classification + segmentation
- 🧠 Optional router module for task-aware path selection
- 🩺 Medical-grade preprocessing & loss design
- ⚖️ Focal + Dice loss for class imbalance & region accuracy
- 🔁 Configurable training & evaluation pipeline
- 🎛️ Streamlit UI for image upload + visual feedback

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
torch
torchvision
numpy
scikit-learn
tqdm
Pillow
streamlit
matplotlib
```

---

## 📊 Logging and Evaluation

- Logs saved to `logs/training.log`
- Metric plots auto-saved after training
- Results saved to `logs/metrics_log.json`
- Final model: `models/mtl_model_final.pth`



## 👤 Author

**Dr. Balmukund Mishra**  
AI Researcher | Medical Imaging | Computer Vision  
📬 balmukundmishra@example.com
