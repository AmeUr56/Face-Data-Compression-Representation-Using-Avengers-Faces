# 🧠 Face Data Compression & Representation Using Avengers Faces
This project explores the compression and reconstruction of facial data using the <a href='https://www.kaggle.com/datasets/yasserh/avengers-faces-dataset'>**Avengers Faces dataset**</a>. By applying and comparing multiple **dimensionality reduction techniques**.

---

## 📦 Dataset
We use the **Avengers Faces Dataset**, which contains cropped facial images of characters from the Marvel Avengers universe.

---

## 🎯 Objectives
- 🧩 Reduce the dimensionality of face data using multiple techniques
- 🔍 Visualize compressed embeddings in 2D and 3D
- 🔁 Reconstruct faces from compressed representations
- 📊 Compare performance across reduction methods

---

## 🔧 Methods Used
We experiment with both linear and nonlinear dimensionality reduction techniques:
- **PCA and its Variants**
    - **Linear PCA**
    - **Randomized PCA**
    - **Kernel PCA**
- **Random Projection**
- **Locally Linear Embedding**
- **Multi Dimentional Scaling**
- **Isomap**
- **TSNE**

---

## 🧪 Experiments

### ✅ Preprocessing
- Image resizing to fixed dimensions (e.g. 64x64)
- Normalization of pixel values

### ✅ Compression
- Apply each algorithm to compress images into low-dimensional vectors (e.g. 2D, 10D, 50D)

### ✅ Reconstruction
- Rebuild faces from compressed features (PCA and autoencoder only)
- Compute reconstruction error (MSE, PSNR)

### ✅ Evaluation
- Qualitative comparison via visual inspection
- Quantitative via reconstruction metrics

### ✅ Visualization
- Plot compressed data in 2D using matplotlib and Seaborn
- Color-code embeddings based on character identities

---

# 🌐 Streamlit App
https://face-data-compression-representation.streamlit.app

- Upload a face image (or choose from sample Avengers faces)
- Select a dimensionality reduction technique(compression method)
- Choose number of dimensions (compression level)

## 💫 Face Compression & Reconstruction
- See original vs reconstructed face side by side

## 📊 Face 2D and 3D Visualization
- See 2D/3D visualizations of embeddings

---

# 📜 License
This project is licensed under the **MIT License**. Copyright (c) 2025 Ameur.