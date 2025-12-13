# Dataset Inference Attack - Technical Explanation

## Overview

This solution implements a **Dataset Inference Attack** to determine whether specific data subsets were used to train a machine learning model. The attack achieves this by analyzing how the model responds to queries - models tend to be more confident on data they were trained on.

---

## Core Concepts

### 1. Dataset Inference vs Membership Inference

| Aspect | Membership Inference (MI) | Dataset Inference (DI) |
|--------|---------------------------|------------------------|
| **Goal** | Determine if a single sample was in training | Determine if a dataset/subset was in training |
| **Granularity** | Per-sample decision | Aggregate over multiple samples |
| **Success Rate** | Decreases with training set size | Independent of training set size |
| **Key Insight** | Overfitting on individuals | Aggregate statistical patterns |

**Why DI works better**: Even when individual sample MI has ~50% accuracy (coin flip), aggregating weak signals over 100 samples amplifies the detection capability significantly.

---

### 2. LiRA (Likelihood Ratio Attack)

**Source**: "Membership Inference Attacks From First Principles" (Carlini et al., 2022)

#### The Log-Odds Transformation

Instead of using raw probability `p`, LiRA uses:

```
logit(p) = log(p / (1-p))
```

This transforms:
- `p ∈ [0, 1]` → `logit ∈ [-∞, +∞]`

#### Why Log-Odds is Superior

| Probability (p) | Log-Odds | Behavior |
|-----------------|----------|----------|
| 0.01 | -4.60 | Very uncertain (wrong class) |
| 0.10 | -2.20 | Uncertain |
| 0.50 | 0.00 | Maximum uncertainty |
| 0.90 | +2.20 | Confident |
| 0.99 | +4.60 | Very confident |
| 0.999 | +6.91 | Extremely confident |

**Key advantage**: Log-odds amplifies differences at the tails (very high or very low confidence), which is exactly where members and non-members differ most.

#### Implementation

```python
correct_prob = probs[idx, labels]  # Probability of correct class
correct_prob = correct_prob.clamp(min=1e-7, max=1-1e-7)  # Numerical stability
log_odds = torch.log(correct_prob / (1 - correct_prob))
```

---

### 3. Temperature Scaling

Temperature scaling modifies the "sharpness" of the probability distribution by dividing logits before softmax:

```python
scaled_logits = logits / temperature
probs = softmax(scaled_logits)
```

#### Temperature Effects

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| T < 1 (e.g., 0.5) | Sharper (more confident) | Amplifies high-confidence predictions |
| T = 1 | Original distribution | Baseline |
| T > 1 (e.g., 2.0) | Softer (less confident) | Reveals uncertainty patterns |

#### Why Use Multiple Temperatures?

Different temperatures reveal different aspects of model confidence:
- **T=0.25, 0.5**: Amplify subtle confidence differences
- **T=1.0**: Standard behavior
- **T=2.0**: How the model behaves when "smoothed"

Members vs non-members may show different patterns at different temperatures.

#### Implementation

```python
temperatures = [0.25, 0.5, 1.0, 2.0]

for temp in temperatures:
    probs_t = F.softmax(logits / temp, dim=1)
    correct_prob_t = probs_t[idx, labels]
    log_odds_t = torch.log(correct_prob_t / (1 - correct_prob_t))
```

---

### 4. Prediction Margin

The margin is the difference between the logit of the correct class and the second-best class:

```
margin = logit(correct_class) - logit(second_best_class)
```

#### Why Margin Matters

- **Positive margin**: Model predicts correctly
- **Negative margin**: Model predicts incorrectly
- **Large positive margin**: Model is very confident about correct class

**Key insight**: Training pushes decision boundaries away from training points, so members have larger margins.

#### Implementation

```python
correct_logits = logits[idx, labels]
logits_masked = logits.clone()
logits_masked[idx, labels] = float('-inf')  # Mask correct class
second_best = logits_masked.max(dim=1).values
margin = correct_logits - second_best
```

---

### 5. Cross-Entropy Loss

The loss measures how "surprised" the model is by the correct answer:

```
loss = -log(p_correct)
```

#### Member vs Non-Member Loss Distribution

| Statistic | Members | Non-Members |
|-----------|---------|-------------|
| Mean Loss | Lower | Higher |
| Max Loss | Bounded | Unbounded (outliers) |
| Loss Variance | Lower (consistent) | Higher (inconsistent) |

**Critical insight**: Since ALL 100 samples in a member subset were trained on, even the worst-case (max) loss is bounded. Non-members can have outlier samples with very high loss.

---

## Signal Aggregation Strategy

### Why Aggregate Multiple Signals?

No single signal perfectly separates members from non-members. By combining multiple complementary signals, we get a more robust score.

### Our Signal Categories

#### 1. LiRA Log-Odds (Primary)
```python
score += mean_log_odds * 0.30      # Average confidence
score += min_log_odds * 0.12       # Worst-case confidence
score += percentile_5_log_odds * 0.08
score += percentile_10_log_odds * 0.05
score += -std_log_odds * 0.05      # Lower variance = member
```

#### 2. Temperature Ensemble
```python
score += mean_log_odds_low_t * 0.10   # T=0.5
score += min_log_odds_low_t * 0.05
score += mean_log_odds_high_t * 0.10  # T=2.0
score += min_log_odds_high_t * 0.05
score += mean_log_odds_very_low_t * 0.05  # T=0.25
```

#### 3. Loss Signals
```python
score += -mean_loss * 0.6          # Lower loss = member
score += -max_loss * 0.10          # Bounded worst-case = member
score += -loss_95th * 0.05         # Near-worst-case
score += -std_loss * 0.08          # Lower variance = member
```

#### 4. Confidence Signals
```python
score += mean_conf * 1.0           # Higher confidence = member
score += min_conf * 0.5            # Worst-case bounded = member
score += conf_5th * 0.3            # 5th percentile
```

#### 5. Margin Signals
```python
score += mean_margin * 0.06
score += min_margin * 0.04
score += margin_5th * 0.03
```

#### 6. Accuracy
```python
score += accuracy * 0.6            # Higher accuracy = member
```

---

## Statistical Insights

### Why Worst-Case Statistics Matter for TPR@FPR=5%

The evaluation metric is **TPR@FPR=5%**, meaning:
- We can only have 5% false positives (25 out of 500 non-members)
- We want to maximize true positives (identify as many of 500 members as possible)

**Key insight**: At the decision threshold, what separates the top non-members from members is the **worst-case behavior**. Non-members have outlier samples that:
- Have very high loss
- Have very low confidence
- Have negative margins (misclassified)

Members have ALL 100 samples trained on, so their worst-case is bounded.

### Percentile-Based Statistics

Using percentiles (5th, 10th, 95th) instead of just min/max:
- More robust to single outliers
- Captures the "near-worst-case" behavior
- Provides smoother signal

```python
sorted_loss = per_sample_loss.sort().values
loss_95th = sorted_loss[94].item()  # 95th percentile (5th worst)
min_loss = sorted_loss[0].item()    # Minimum (best)
max_loss = sorted_loss[99].item()   # Maximum (worst)
```

---

## Model Architecture Details

The target model is a modified ResNet-18:

```python
model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=3, bias=False)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(model.fc.in_features, 9)  # 9 classes
)
```

**Key modifications**:
- First conv layer: kernel_size=5 (instead of 7), stride=1 (instead of 2)
- Final layer: 9 output classes (likely Fashion-MNIST or similar)
- Dropout: 0.2 (important to set `model.eval()` to disable during inference)

---

## Data Format

Each subset contains:
- `images`: Tensor of shape `(100, 1, 28, 28)` - 100 grayscale 28×28 images
- `labels`: Tensor of shape `(100,)` - class labels (0-8)
- `subset_id`: Integer identifier (0-999)

The images are converted to RGB by repeating the single channel:
```python
if images.shape[1] == 1:
    images = images.repeat(1, 3, 1, 1)  # (100, 1, 28, 28) → (100, 3, 28, 28)
```

---

## Evaluation Metric: TPR@FPR=5%

### Definition

- **TPR (True Positive Rate)** = TP / (TP + FN) = Recall
- **FPR (False Positive Rate)** = FP / (FP + TN)

**TPR@FPR=5%** means: "What percentage of true members do we correctly identify when we set the threshold such that only 5% of non-members are incorrectly flagged?"

### In This Task

- 1000 subsets total
- 500 members (training data)
- 500 non-members (not training data)
- At FPR=5%: We allow 25 false positives (non-members incorrectly labeled as members)
- Goal: Maximize true positives (correctly identified members)

### Why This Metric?

It tests whether the attack can confidently identify members without many false alarms - important for real-world ownership claims where false accusations are costly.

---

## References

1. **Dataset Inference**: Maini, Yaghini, Papernot. "Dataset Inference: Ownership Resolution in Machine Learning." ICLR 2021.

2. **LiRA**: Carlini et al. "Membership Inference Attacks From First Principles." IEEE S&P 2022.

3. **Temperature Scaling**: Guo et al. "On Calibration of Modern Neural Networks." ICML 2017.

---

## Summary

Our attack combines:
1. **LiRA log-odds** - State-of-the-art confidence transformation
2. **Temperature scaling** - Multi-scale analysis of model confidence
3. **Worst-case statistics** - Critical for TPR@FPR metric
4. **Multiple complementary signals** - Loss, confidence, margin, accuracy
5. **Aggregation over 100 samples** - Amplifies weak individual signals

The key insight is that **members have bounded worst-case behavior** because all 100 samples were seen during training, while **non-members have unbounded outliers**.
