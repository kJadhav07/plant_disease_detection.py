"""
Plant Disease Detection System
==============================
A beginner-friendly project (Botany + Data Science + Machine Learning).

What it does
------------
1. Loads a leaf image and shows it.
2. Analyses leaf colour and finds spots / damaged areas with OpenCV.
3. Extracts numeric features from the image.
4. Uses a Random Forest model (trained on YOUR labelled dataset) to
   classify the leaf.
5. Shows a colour-distribution graph and saves results to a CSV file.

IMPORTANT
---------
Colour analysis alone can NOT diagnose a plant disease. Brown spots, for
example, can come from disease, sunburn, pests, nutrient problems or
natural ageing. The spot detection here is only a visual hint. The final
"status" comes from the machine-learning model, and it is only as good as
the dataset it was trained on. For real crop problems, ask a plant
pathologist or an agricultural extension office.

Dataset layout (folder names are the labels)
--------------------------------------------
dataset/
    healthy/     <- images of healthy leaves
    diseased/    <- images of diseased leaves

You can also use a multi-class dataset such as PlantVillage, e.g.
dataset/Tomato___healthy/, dataset/Tomato___Early_blight/ ...
Any folder whose name contains the word "healthy" is treated as healthy;
every other folder is treated as a disease category.

Install:  pip install opencv-python numpy matplotlib scikit-learn
Run:      python plant_disease_detection.py
"""

import csv
import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Settings (easy to change)
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "analysis_results.csv")

IMAGE_SIZE = (128, 128)       # every image is resized to this size
MAX_IMAGES_PER_CLASS = 300    # limit per folder so training stays fast
MIN_SPOT_AREA = 15            # ignore specks smaller than this (pixels)
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Colour ranges in HSV (OpenCV: H 0-179, S 0-255, V 0-255)
COLOUR_RANGES = {
    "Green":  ((36, 40, 40), (85, 255, 255)),
    "Yellow": ((20, 70, 100), (35, 255, 255)),
    "Brown":  ((5, 60, 30), (19, 255, 200)),
}
DARK_V_MAX = 50  # pixels darker than this inside the leaf count as "Black"


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def load_image(path):
    """Read an image from disk and return it (BGR, resized).
    Raises an error with a clear message if the file cannot be used."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not path.lower().endswith(VALID_EXTENSIONS):
        raise ValueError("Unsupported file type. Use jpg, jpeg, png, bmp or tif.")
    # np.fromfile + imdecode also works for paths with special characters
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The file could not be read as an image (maybe corrupted).")
    return cv2.resize(image, IMAGE_SIZE)


def get_leaf_mask(image_bgr):
    """Find which pixels belong to the leaf (white = leaf, black = background).

    Method: the leaf is usually more colourful than the background, so we
    threshold the saturation channel (Otsu), then fill the biggest shape.
    Filling the shape keeps dark spots INSIDE the leaf as part of the leaf.
    If nothing sensible is found we simply use the whole image."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [biggest], -1, 255, thickness=cv2.FILLED)

    leaf_fraction = np.count_nonzero(filled) / filled.size
    if leaf_fraction < 0.05:                 # leaf not found -> use whole image
        filled[:] = 255
    return filled


def colour_percentages(image_bgr, leaf_mask):
    """Return (dict of colour -> % of leaf pixels, dict of boolean masks)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    leaf_pixels = max(np.count_nonzero(leaf_mask), 1)

    masks = {}
    for name, (low, high) in COLOUR_RANGES.items():
        found = cv2.inRange(hsv, np.array(low), np.array(high))
        masks[name] = cv2.bitwise_and(found, leaf_mask)

    dark = (hsv[:, :, 2] < DARK_V_MAX).astype(np.uint8) * 255
    masks["Black"] = cv2.bitwise_and(dark, leaf_mask)

    percentages = {n: 100.0 * np.count_nonzero(m) / leaf_pixels for n, m in masks.items()}
    percentages["Other"] = max(0.0, 100.0 - sum(percentages.values()))
    return percentages, masks


def find_spots(masks):
    """Count spots (small coloured patches) of each spot colour.
    Returns (counts per colour, combined spot mask)."""
    counts = {}
    combined = np.zeros_like(masks["Brown"])
    for name in ("Yellow", "Brown", "Black"):
        contours, _ = cv2.findContours(masks[name], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        counts[name] = sum(1 for c in contours if cv2.contourArea(c) >= MIN_SPOT_AREA)
        combined = cv2.bitwise_or(combined, masks[name])
    return counts, combined


def extract_features(image_bgr):
    """Turn an image into a list of numbers (features) for the ML model.

    Features: colour percentages, mean/std of H, S, V, a 16-bin hue
    histogram, number of spots and the percentage of 'damaged' area."""
    leaf_mask = get_leaf_mask(image_bgr)
    percentages, masks = colour_percentages(image_bgr, leaf_mask)
    spot_counts, spot_mask = find_spots(masks)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    leaf_pixels = hsv[leaf_mask > 0]                      # shape (N, 3)
    means = leaf_pixels.mean(axis=0)
    stds = leaf_pixels.std(axis=0)
    hue_hist, _ = np.histogram(leaf_pixels[:, 0], bins=16, range=(0, 180))
    hue_hist = hue_hist / max(hue_hist.sum(), 1)

    damaged = 100.0 * np.count_nonzero(spot_mask) / max(np.count_nonzero(leaf_mask), 1)

    features = [percentages["Green"], percentages["Yellow"], percentages["Brown"],
                percentages["Black"], percentages["Other"], damaged,
                spot_counts["Yellow"], spot_counts["Brown"], spot_counts["Black"]]
    features += list(means) + list(stds) + list(hue_hist)

    details = {
        "percentages": percentages,
        "spot_counts": spot_counts,
        "damaged_percent": damaged,
        "leaf_mask": leaf_mask,
        "spot_mask": spot_mask,
    }
    return np.array(features, dtype=float), details


def is_healthy_label(label):
    """A folder/label counts as healthy if its name contains 'healthy'."""
    return "healthy" in label.lower()


# ----------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------
class DiseaseModel:
    """Trains and uses a Random Forest to classify leaf images."""

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.accuracy = None

    def _collect_images(self):
        """Walk through the dataset folder and return a list of
        (image_path, label). The label is the name of the image's folder."""
        samples = {}
        for folder, _, files in os.walk(DATASET_DIR):
            images = sorted(f for f in files if f.lower().endswith(VALID_EXTENSIONS))
            if images:
                label = os.path.basename(folder)
                paths = [os.path.join(folder, f) for f in images[:MAX_IMAGES_PER_CLASS]]
                samples.setdefault(label, []).extend(paths)
        return samples

    def train(self):
        """Load the dataset, split it 80/20, train the model and compute
        accuracy on the unseen 20%. Returns True if training worked."""
        if not os.path.isdir(DATASET_DIR):
            print(f"\n[!] Dataset folder not found: {DATASET_DIR}")
            return False

        samples = self._collect_images()
        if len(samples) < 2:
            print("\n[!] The dataset needs at least 2 labelled folders "
                  "(for example 'healthy' and 'diseased') containing images.")
            return False

        print("\nLoading dataset and extracting features (this can take a while)...")
        features, labels = [], []
        for label, paths in samples.items():
            used = 0
            for path in paths:
                try:
                    vector, _ = extract_features(load_image(path))
                    features.append(vector)
                    labels.append(label)
                    used += 1
                except Exception:
                    continue          # skip unreadable images
            print(f"  {label}: {used} images used")

        if len(set(labels)) < 2 or len(labels) < 10:
            print("\n[!] Not enough readable images to train (need at least 10 "
                  "images across 2 classes).")
            return False

        x = np.array(features)
        y = np.array(labels)
        counts = {c: labels.count(c) for c in set(labels)}
        can_stratify = min(counts.values()) >= 2
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42,
            stratify=y if can_stratify else None)

        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_test)
        self.accuracy = accuracy_score(y_test, predictions)
        self.is_trained = True

        print(f"\nModel trained on {len(x_train)} images, tested on {len(x_test)} images.")
        print(f"Test accuracy: {self.accuracy * 100:.1f}%")
        print(classification_report(y_test, predictions, zero_division=0))
        print("Note: accuracy on a dataset does not guarantee the same accuracy "
              "on your own photos (lighting and background matter a lot).")
        return True

    def predict(self, feature_vector):
        """Predict the category of one image. Returns (label, confidence %)."""
        if not self.is_trained:
            raise RuntimeError("The model is not trained yet.")
        vector = feature_vector.reshape(1, -1)
        label = self.model.predict(vector)[0]
        confidence = self.model.predict_proba(vector).max() * 100
        return str(label), float(confidence)


class LeafAnalyser:
    """Holds one leaf image and its analysis results."""

    def __init__(self):
        self.path = None
        self.image = None
        self.details = None
        self.result = None

    def load(self, path):
        """Load an image and forget any older analysis."""
        self.image = load_image(path)
        self.path = path
        self.details = None
        self.result = None

    def show_image(self):
        """Display the loaded image in a window."""
        if self.image is None:
            raise RuntimeError("No image loaded. Choose option 1 first.")
        plt.figure("Leaf Image")
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(self.path))
        plt.axis("off")
        plt.show()

    def analyse(self, disease_model):
        """Run colour analysis, spot detection and ML prediction."""
        if self.image is None:
            raise RuntimeError("No image loaded. Choose option 1 first.")

        features, self.details = extract_features(self.image)
        pct = self.details["percentages"]
        spots = self.details["spot_counts"]

        # Dominant colour = the colour covering most of the leaf
        dominant = max(("Green", "Yellow", "Brown", "Black", "Other"), key=lambda c: pct[c])

        # Spot colours: only report colours with a noticeable number of spots
        spot_colours = [c for c in ("Yellow", "Brown", "Black") if spots[c] >= 2]
        spots_text = ", ".join(spot_colours) if spot_colours else "None clearly detected"

        # Status comes from the ML model, NOT from spots alone
        if disease_model.is_trained:
            label, confidence = disease_model.predict(features)
            status = "Likely Healthy" if is_healthy_label(label) else "Potentially Diseased"
            prediction = f"{label} ({confidence:.0f}% model confidence)"
        else:
            status = "Unknown (no trained model - add a dataset)"
            prediction = "Not available"

        self.result = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": os.path.basename(self.path),
            "Leaf Status": status,
            "Dominant Colour": dominant,
            "Detected Spots": spots_text,
            "Spot Count": sum(spots.values()),
            "Damaged Area %": round(self.details["damaged_percent"], 1),
            "Model Prediction": prediction,
        }

    def print_result(self):
        """Print the result in a readable format."""
        if self.result is None:
            raise RuntimeError("Nothing analysed yet. Choose option 2 first.")
        r = self.result
        print("\nPlant Analysis Result:")
        print(f"  * Image: {r['Image']}")
        print(f"  * Leaf Status: {r['Leaf Status']}")
        print(f"  * Dominant Colour: {r['Dominant Colour']}")
        print(f"  * Detected Spots: {r['Detected Spots']}")
        print(f"  * Damaged Area: {r['Damaged Area %']}%")
        print(f"  * Model Prediction: {r['Model Prediction']}")
        print("\n  (This is a screening aid, not a diagnosis. Spots can have many "
              "causes besides disease.)")

    def show_colour_graph(self):
        """Show a bar chart of leaf colours and a hue histogram."""
        if self.details is None:
            raise RuntimeError("Nothing analysed yet. Choose option 2 first.")
        pct = self.details["percentages"]
        names = list(pct.keys())
        values = [pct[n] for n in names]
        bar_colours = ["green", "gold", "saddlebrown", "black", "gray"]

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hues = hsv[:, :, 0][self.details["leaf_mask"] > 0]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), num="Leaf Colour Graph")
        axes[0].bar(names, values, color=bar_colours)
        axes[0].set_title("Leaf colour distribution")
        axes[0].set_ylabel("% of leaf area")
        axes[1].hist(hues, bins=36, range=(0, 180), color="seagreen")
        axes[1].set_title("Hue histogram (0-179)")
        axes[1].set_xlabel("Hue value")
        axes[1].set_ylabel("Pixels")
        fig.tight_layout()
        plt.show()

    def save_result(self):
        """Append the latest result to results/analysis_results.csv."""
        if self.result is None:
            raise RuntimeError("Nothing analysed yet. Choose option 2 first.")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        new_file = not os.path.isfile(CSV_PATH)
        with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.result.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(self.result)
        print(f"\nResult saved to: {CSV_PATH}")


class PlantDiseaseApp:
    """The text menu that connects everything together."""

    def __init__(self):
        self.analyser = LeafAnalyser()
        self.disease_model = DiseaseModel()

    def setup(self):
        """Create folders and train the model once at start-up."""
        os.makedirs(DATASET_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        if not self.disease_model.train():
            print("\nYou can still use the colour analysis, but the model "
                  "prediction will be unavailable.")
            print("Put labelled images in dataset/healthy/ and dataset/diseased/ "
                  "(for example from the PlantVillage dataset) and restart.")

    def upload_image(self):
        """Ask for an image path and load it."""
        path = input("Enter the full path of the leaf image: ").strip().strip('"').strip("'")
        self.analyser.load(os.path.expanduser(path))
        print("Image loaded successfully.")
        try:
            self.analyser.show_image()
        except Exception as error:   # e.g. no display available
            print(f"(Could not display the image: {error})")

    def run(self):
        """Main loop: show the menu until the user chooses Exit."""
        print("Welcome to Plant Disease Detection System!")
        self.setup()

        while True:
            print("\n1. Upload Leaf Image")
            print("2. Analyse Leaf")
            print("3. Show Detection Result")
            print("4. Display Leaf Colour Graph")
            print("5. Save Results")
            print("6. Exit")
            choice = input("Choose an option (1-6): ").strip()

            try:
                if choice == "1":
                    self.upload_image()
                elif choice == "2":
                    self.analyser.analyse(self.disease_model)
                    print("Analysis complete. Choose option 3 to see the result.")
                elif choice == "3":
                    self.analyser.print_result()
                elif choice == "4":
                    self.analyser.show_colour_graph()
                elif choice == "5":
                    self.analyser.save_result()
                elif choice == "6":
                    print("Goodbye!")
                    break
                else:
                    print("Invalid choice. Please enter a number from 1 to 6.")
            except (FileNotFoundError, ValueError, RuntimeError) as error:
                print(f"\n[Error] {error}")
            except Exception as error:   # last safety net so the program never crashes
                print(f"\n[Unexpected error] {error}")


def main():
    """Program entry point."""
    try:
        PlantDiseaseApp().run()
    except (KeyboardInterrupt, EOFError):
        print("\nProgram stopped. Goodbye!")


if __name__ == "__main__":
    main()
