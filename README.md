# Smart Product Pricing Solution 

**Members:** [Jiya Sinha](https://github.com/sinhajiya), [Himadri Sharma](https://github.com/HIMADRI121), [Rashi Bharti](https://github.com/rashibharti28)

---

## Highlights

* **Approach Type:** Multimodal (Text + Image)
* **Overview:**

  * Fine-tuned **SigLIP** text projection head and final transformer layer.
  * **Frozen image encoder** to preserve visual generalization.
  * **Weighted fusion** of image and text embeddings (0.3 × image + 0.7 × text).
  * **3-layer MLP regression head** for log-price prediction.
 * **Best Validation SMAPE:** **47%**

---

##  Methodology

### 1. Observations

* Product **text descriptions** were more informative than images.
* **Images** often lacked rich cues (mostly packaging appearance).
* Text data captured **semantic details** — type, brand, quantity — crucial for pricing.
* **Price distribution** was skewed (fewer high-priced products).

### 2. Solution Strategy

* Treated as a **multimodal regression task** using both text and image embeddings.
* **Log-transformed prices** to stabilize regression performance.
* Evaluated on **SMAPE** (Symmetric Mean Absolute Percentage Error) using exponentiated predictions.

---

##  Model Architecture

### Components

| Component           | Description                                                                         |
| ------------------- | ----------------------------------------------------------------------------------- |
| **Image Encoder**   | Frozen SigLIP vision tower                                                          |
| **Text Encoder**    | SigLIP text tower (last transformer layer + projection head fine-tuned)             |
| **Fusion Layer**    | Weighted average of normalized image and text embeddings (0.3 × image + 0.7 × text) |
| **Regression Head** | 3-layer MLP with GELU activations and dropout                                       |

**Figure: Model Overview**

<img width="3960" height="2860" alt="image" src="https://github.com/user-attachments/assets/861b20e2-1693-44bf-b3ad-ee9442fb8d05" />

---

## Data Processing

###  Text Processing

1. Convert to lowercase
2. Remove URLs and redundant punctuation
3. Normalize spaces, remove non-alphanumeric chars
4. Extract and merge structured fields (name, description, bullet points, quantity)
5. Remove repetitive phrases

### Image Processing

1. Download images from URLs
2. Resize uniformly to **128×128** 
3. Save in `.jpeg` format with **80% quality**. (Step 2 and 3 done due to limited storage).
4. Maintain consistent filenames using hashed identifiers

---

##  Training Details

* **Loss Function:** L1 Loss on log(price)
* **Evaluation Metric:** SMAPE on exponentiated price scale
* **Optimizer:** AdamW 
* **Regularization:** Dropout in MLP

---

## Results

| Metric               | Score   |
| -------------------- | ------- |
| **Validation SMAPE** | **47%** |

---

##  Future Work

* Explore **adaptive fusion** strategies (learned weighting).
* Fine-tune **visual encoder** on specific product categories.

