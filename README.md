<div align="center">

![Plant Disease Detection System](images/banner.png)

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikitlearn&logoColor=white)
![Level](https://img.shields.io/badge/Level-Beginner%20Friendly-2ea44f)

**Spot the signs. Let machine learning help you decide.**

</div>

---

## 📖 About the Project

The **Plant Disease Detection System** analyses a photo of a plant leaf and tells you whether it looks **healthy** or **potentially diseased**. It combines:

- 🎨 **Image processing (OpenCV)** to measure leaf colours and find spots or damaged areas
- 🧮 **Feature extraction (NumPy)** to turn each image into numbers
- 🌲 **A Random Forest model (scikit-learn)** trained on real, labelled leaf images
- 📈 **Graphs (Matplotlib)** to show the leaf colour distribution

> ⚠️ **Important:** Colour analysis alone cannot diagnose a plant disease. Brown spots can also come from sunburn, pests, nutrient problems or natural ageing. The final status comes from the trained model, and the result is a **screening aid, not a diagnosis**.

---

## ✨ Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | 📂 Load leaf image | Choose any `.jpg`, `.png`, `.bmp` or `.tif` file |
| 2 | 🖼️ Display image | Preview the leaf before analysis |
| 3 | 🎨 Colour analysis | Green, yellow, brown, black and other % of the leaf |
| 4 | 🔍 Spot detection | Counts yellow, brown and black spots; estimates damaged area |
| 5 | 🧬 Feature extraction | Colour %, HSV statistics, hue histogram, spot counts |
| 6 | 🤖 ML classification | Healthy vs. diseased category, with model confidence |
| 7 | 📋 Result display | Clear text summary of the analysis |
| 8 | 📊 Colour graph | Bar chart + hue histogram |
| 9 | 💾 Save to CSV | Every analysis is stored in `results/` |

---

## 🗂️ Project Structure

```
plant-disease-detection/
│
├── plant_disease_detection.py   # 🧠 Main program
├── README.md                    # 📘 Documentation
├── images/                      # 🖼️ Pictures used in this README
│
├── dataset/                     # 🌿 Labelled leaf images (you add these)
│   ├── healthy/
│   └── diseased/
│
└── results/                     # 💾 Saved analysis results
    └── analysis_results.csv     #    (created automatically)
```

---

## 🚀 Getting Started

### 1️⃣ Install the libraries

```bash
pip install opencv-python numpy matplotlib scikit-learn
```

### 2️⃣ Add a real labelled dataset

Put leaf images into labelled folders inside `dataset/`.

**Simple (two classes):**

```
dataset/healthy/    ← healthy leaf images
dataset/diseased/   ← diseased leaf images
```

**Multi-class (e.g. PlantVillage from Kaggle):**

```
dataset/Tomato___healthy/
dataset/Tomato___Early_blight/
dataset/Tomato___Late_blight/
...
```

> 💡 Any folder whose name contains the word **"healthy"** is treated as healthy. Every other folder is treated as a disease category.

### 3️⃣ Run the program

```bash
python plant_disease_detection.py
```

---

## 🖥️ How to Use

```
Welcome to Plant Disease Detection System!

1. Upload Leaf Image
2. Analyse Leaf
3. Show Detection Result
4. Display Leaf Colour Graph
5. Save Results
6. Exit
```

**Typical workflow:** `1` → `2` → `3` → `4` → `5`

<div align="center">

![Menu](images/menu.png)

</div>

### 🔬 Example Analysis

![Example analysis](images/analysis_example.png)

> 🖼️ The images in this README are generated illustrations (a drawn sample leaf and a mock-up terminal), not real photos of diseased plants. Use a real dataset for actual results.

### 📋 Example Output

```
Plant Analysis Result:
  * Image: leaf.jpg
  * Leaf Status: Potentially Diseased
  * Dominant Colour: Green
  * Detected Spots: Brown
  * Damaged Area: 3.0%
  * Model Prediction: Tomato___Early_blight (92% model confidence)
```

---

## 🧠 How It Works

![Workflow](images/workflow.png)

1. **Training:** at start-up the dataset is split **80% training / 20% testing**. The model's accuracy on the unseen 20% is printed.
2. **Analysis:** the new image goes through the same feature extraction.
3. **Prediction:** the model predicts the category. Spots are shown as extra information and are **never** used alone to declare a disease.

---

## 📚 Concepts Used

- 🐍 Python, functions, loops, conditionals
- 🏗️ Object-Oriented Programming (classes and objects)
- 🛡️ Exception handling
- 👁️ OpenCV (colour spaces, masks, contours)
- 🔢 NumPy
- 📉 Matplotlib
- 🤖 Scikit-learn (Random Forest, train/test split, accuracy)

---

## ⚠️ Limitations

- 📸 Accuracy on a dataset does **not** guarantee the same accuracy on your own photos. Lighting, background and camera quality matter a lot.
- 🍃 The model only knows the diseases and plants in the dataset you train it with.
- 🔬 This tool is for **learning and screening only**. For real crop problems, consult a plant pathologist or an agricultural extension office.

---

## 🔮 Ideas for Improvement

- Add a deep learning model (CNN) for higher accuracy
- Build a simple GUI with Tkinter
- Save and reload the trained model to skip retraining
- Add treatment suggestions for each disease category

---

<div align="center">

Made with 💚 for plant lovers and Python beginners

</div>
