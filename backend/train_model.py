from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TOKENIZED_DATASET_PATH = (
    PROJECT_ROOT / "dataset" / "tokenized_pytorch"
)

OUTPUT_MODEL_PATH = (
    PROJECT_ROOT / "model" / "muril_sentiment_model"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "google/muril-base-cased"

NUM_LABELS = 3

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 8

EPOCHS = 3

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

print("\nChecking hardware...")

if torch.cuda.is_available():

    device = torch.device("cuda")

    print(
        "GPU available:",
        torch.cuda.get_device_name(0)
    )

else:

    device = torch.device("cpu")

    print("GPU not available.")

    print(
        "Training on CPU may be very slow."
    )


# ============================================================
# PYTORCH DATASET
# ============================================================

class SentimentDataset(Dataset):

    def __init__(self, file_path):

        print(f"\nLoading dataset:")
        print(file_path)

        data = torch.load(
            file_path,
            map_location="cpu",
            weights_only=True
        )

        self.input_ids = data["input_ids"]

        self.attention_mask = data["attention_mask"]

        self.labels = data["labels"]

        print(
            "Examples:",
            len(self.labels)
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids": self.input_ids[index],

            "attention_mask": self.attention_mask[index],

            "labels": self.labels[index]
        }


# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print("LOADING DATA")
print("========================================")


train_dataset = SentimentDataset(
    TOKENIZED_DATASET_PATH / "train.pt"
)

validation_dataset = SentimentDataset(
    TOKENIZED_DATASET_PATH / "validation.pt"
)

test_dataset = SentimentDataset(
    TOKENIZED_DATASET_PATH / "test.pt"
)


print("\nDataset sizes:")

print(
    "Training:",
    len(train_dataset)
)

print(
    "Validation:",
    len(validation_dataset)
)

print(
    "Testing:",
    len(test_dataset)
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n========================================")
print("LOADING MuRIL TOKENIZER")
print("========================================")


tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZED_DATASET_PATH,
    use_fast=False,
    local_files_only=True
)


# ============================================================
# LOAD MuRIL MODEL
# ============================================================

print("\n========================================")
print("LOADING MuRIL MODEL")
print("========================================")


model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=NUM_LABELS,

    id2label=ID_TO_LABEL,

    label2id=LABEL_TO_ID
)


model.to(device)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    all_predictions = []

    all_labels = []


    for batch_number, batch in enumerate(
        train_loader,
        start=1
    ):

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)


        # Clear old gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            labels=labels

        )


        loss = outputs.loss

        logits = outputs.logits


        # Backpropagation

        loss.backward()


        # Prevent exploding gradients

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM
        )


        # Update weights

        optimizer.step()


        # Metrics

        total_loss += loss.item()


        predictions = torch.argmax(
            logits,
            dim=-1
        )


        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        all_labels.extend(
            labels.detach().cpu().tolist()
        )


        # Print progress

        if batch_number % 25 == 0:

            print(
                f"Batch "
                f"{batch_number}/{len(train_loader)} "
                f"| Loss: {loss.item():.4f}"
            )


    average_loss = (
        total_loss / len(train_loader)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return average_loss, accuracy


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(data_loader):

    model.eval()

    total_loss = 0.0

    all_predictions = []

    all_labels = []


    with torch.no_grad():

        for batch in data_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["labels"].to(device)


            outputs = model(

                input_ids=input_ids,

                attention_mask=attention_mask,

                labels=labels

            )


            loss = outputs.loss

            logits = outputs.logits


            total_loss += loss.item()


            predictions = torch.argmax(
                logits,
                dim=-1
            )


            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )


    average_loss = (
        total_loss / len(data_loader)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(

            all_labels,

            all_predictions,

            average="weighted",

            zero_division=0

        )
    )


    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions
    )


# ============================================================
# TRAINING
# ============================================================

print("\n========================================")
print("STARTING MuRIL FINE-TUNING")
print("========================================")


best_f1 = 0.0


for epoch in range(1, EPOCHS + 1):

    print("\n")
    print("=" * 60)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    print("\nTraining results:")

    print(
        f"Loss: {train_loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        validation_loss,
        validation_accuracy,
        validation_precision,
        validation_recall,
        validation_f1,
        _,
        _
    ) = evaluate(
        validation_loader
    )


    print("\nValidation results:")

    print(
        f"Loss: {validation_loss:.4f}"
    )

    print(
        f"Accuracy: "
        f"{validation_accuracy * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{validation_precision * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{validation_recall * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{validation_f1 * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_f1 > best_f1:

        best_f1 = validation_f1

        print(
            "\nNew best model found!"
        )

        OUTPUT_MODEL_PATH.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save_pretrained(
            OUTPUT_MODEL_PATH
        )

        tokenizer.save_pretrained(
            OUTPUT_MODEL_PATH
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n========================================")
print("LOADING BEST MODEL")
print("========================================")


model = AutoModelForSequenceClassification.from_pretrained(
    OUTPUT_MODEL_PATH,
    local_files_only=True
)

model.to(device)


# ============================================================
# FINAL TEST
# ============================================================

print("\n========================================")
print("FINAL TEST EVALUATION")
print("========================================")


(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    true_labels,
    predicted_labels
) = evaluate(
    test_loader
)


print("\nFINAL TEST RESULTS")

print(
    f"Loss: {test_loss:.4f}"
)

print(
    f"Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Precision: "
    f"{test_precision * 100:.2f}%"
)

print(
    f"Recall: "
    f"{test_recall * 100:.2f}%"
)

print(
    f"F1 Score: "
    f"{test_f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")


print(
    classification_report(

        true_labels,

        predicted_labels,

        target_names=[
            "Negative",
            "Neutral",
            "Positive"
        ],

        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")


print(
    confusion_matrix(
        true_labels,
        predicted_labels
    )
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("TRAINING COMPLETED")
print("========================================")


print(
    "\nBest validation F1:",
    round(best_f1 * 100, 2),
    "%"
)


print("\nTrained model saved at:")

print(OUTPUT_MODEL_PATH)