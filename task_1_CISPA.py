import torch
import pandas as pd
import requests
import sys
import torchvision.models as models
import os
import torch.nn.functional as F
# Made by Val with love and claude 4.5
# --------------------------------
# SETUP MODEL
# --------------------------------

print("Setting up model...")
model = models.resnet18(weights=None)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=3, bias=False)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(model.fc.in_features, 9)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model.load_state_dict(torch.load("classifier.pt", map_location=device, weights_only=True))
model.to(device)
model.eval()  # Important: set to eval mode to disable dropout

# --------------------------------
# LOAD DATASET
# --------------------------------

print("Loading dataset...")
dataset = torch.load("subsets_dataset.pt", weights_only=False)
print(f"Number of subsets: {len(dataset)}")

# --------------------------------
# MEMBERSHIP INFERENCE
# --------------------------------

def compute_membership_score(images, labels, model, device):
    """
    Dataset Inference using LiRA + Temperature Scaling + Worst-case analysis.
    
    LiRA: log-odds = log(p / (1-p)) gives better separation
    Temperature: different temps reveal different aspects of confidence
    Worst-case: members have bounded worst samples, non-members have outliers
    """
    images = images.to(device)
    labels = labels.to(device)
    
    # Convert grayscale to RGB
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    
    with torch.inference_mode():
        logits = model(images)
        
        n_samples = len(labels)
        idx = torch.arange(n_samples, device=device)
        
        # ============================================
        # STANDARD METRICS (T=1.0)
        # ============================================
        probs = F.softmax(logits, dim=1)
        correct_prob = probs[idx, labels].clamp(min=1e-7, max=1-1e-7)
        
        # LiRA log-odds
        log_odds = torch.log(correct_prob / (1 - correct_prob))
        sorted_log_odds = log_odds.sort().values
        
        mean_log_odds = log_odds.mean().item()
        min_log_odds = sorted_log_odds[0].item()
        percentile_5_log_odds = sorted_log_odds[4].item()
        
        # Loss
        per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
        mean_loss = per_sample_loss.mean().item()
        max_loss = per_sample_loss.max().item()
        
        # Confidence
        mean_conf = correct_prob.mean().item()
        min_conf = correct_prob.min().item()
        
        # ============================================
        # LOW TEMPERATURE (T=0.5) - Sharper predictions
        # ============================================
        probs_low_t = F.softmax(logits / 0.5, dim=1)
        correct_prob_low_t = probs_low_t[idx, labels].clamp(min=1e-7, max=1-1e-7)
        log_odds_low_t = torch.log(correct_prob_low_t / (1 - correct_prob_low_t))
        
        mean_log_odds_low_t = log_odds_low_t.mean().item()
        min_log_odds_low_t = log_odds_low_t.min().item()
        
        # ============================================
        # HIGH TEMPERATURE (T=2.0) - Softer predictions
        # ============================================
        probs_high_t = F.softmax(logits / 2.0, dim=1)
        correct_prob_high_t = probs_high_t[idx, labels].clamp(min=1e-7, max=1-1e-7)
        log_odds_high_t = torch.log(correct_prob_high_t / (1 - correct_prob_high_t))
        
        mean_log_odds_high_t = log_odds_high_t.mean().item()
        min_log_odds_high_t = log_odds_high_t.min().item()
        
        # ============================================
        # PREDICTION MARGIN
        # ============================================
        correct_logits = logits[idx, labels]
        logits_masked = logits.clone()
        logits_masked[idx, labels] = float('-inf')
        second_best = logits_masked.max(dim=1).values
        margin = correct_logits - second_best
        
        mean_margin = margin.mean().item()
        min_margin = margin.min().item()
        
        # ============================================
        # ACCURACY
        # ============================================
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
        
        # ============================================
        # COMBINED SCORE - Carefully balanced
        # ============================================
        score = 0.0
        
        # Primary: LiRA log-odds at T=1.0 (most important)
        score += mean_log_odds * 0.25
        score += min_log_odds * 0.10
        score += percentile_5_log_odds * 0.05
        
        # Temperature ensemble (secondary)
        score += mean_log_odds_low_t * 0.08
        score += min_log_odds_low_t * 0.04
        score += mean_log_odds_high_t * 0.08
        score += min_log_odds_high_t * 0.04
        
        # Loss (complementary signal)
        score += -mean_loss * 0.5
        score += -max_loss * 0.08
        
        # Confidence (0-1 scale, so higher weights)
        score += mean_conf * 0.8
        score += min_conf * 0.4
        
        # Margin (logit scale)
        score += mean_margin * 0.05
        score += min_margin * 0.03
        
        # Accuracy bonus
        score += accuracy * 0.5
        
    return score

print("Computing membership scores for all subsets...")
subset_ids = []
membership_scores = []

for i in range(1000):
    subset_key = f"subset_{i}"
    subset = dataset[subset_key]
    
    images = subset["images"]
    labels = subset["labels"]
    subset_id = subset["subset_id"]
    
    score = compute_membership_score(images, labels, model, device)
    
    subset_ids.append(subset_id)
    membership_scores.append(score)
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/1000 subsets")

# --------------------------------
# SAVE SUBMISSION
# --------------------------------

submission_df = pd.DataFrame({
    "subset_id": subset_ids,
    "membership": membership_scores
})

# Sort by subset_id to ensure correct order
submission_df = submission_df.sort_values("subset_id").reset_index(drop=True)

output_file = "submission.csv"
submission_df.to_csv(output_file, index=None)
print(f"\nSubmission saved to {output_file}")
print(f"Number of rows: {len(submission_df)}")
print(f"Subset IDs range: {submission_df['subset_id'].min()} - {submission_df['subset_id'].max()}")
print(f"Membership scores range: {submission_df['membership'].min():.4f} - {submission_df['membership'].max():.4f}")

# Preview
print("\nFirst 5 rows:")
print(submission_df.head())

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

BASE_URL  = "http://35.192.205.84:80"
API_KEY   = "9caa40f243393bb4800d686906262257"

TASK_ID   = "06-dataset-inference-vision"
FILE_PATH = output_file

SUBMIT = True  # Set to True to enable submission

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if SUBMIT:
    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                "file": (os.path.basename(FILE_PATH), f, "csv"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120),
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        submission_id = body.get("submission_id")
        print("Successfully submitted.")
        print("Server response:", body)
        if submission_id:
            print(f"Submission ID: {submission_id}")

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)
