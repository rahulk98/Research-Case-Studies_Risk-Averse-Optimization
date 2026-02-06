import argparse
import numpy as np
import cvxpy as cp
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

"""
Simple SVM for cats vs dogs classification
Labels: -1 -> cats, 1 -> dogs
"""

## Solver options
solver_opts = {"eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 10000}


def load_images(dimension=64, load_in_gray=True, max_per_class=None):
    """Load and preprocess images from train directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "train")
    images = []
    labels = []
    class_counts = {"cat": 0, "dog": 0}

    for filename in tqdm(sorted(os.listdir(path)), desc="Loading images"):
        if filename.endswith(".jpg"):
            if max_per_class is not None:
                class_type = "cat" if "cat" in filename else "dog"
                if class_counts[class_type] >= max_per_class:
                    continue

            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = preprocess_image(
                img_array, target_size=(dimension, dimension), to_gray=load_in_gray
            )
            images.append(img_array)
            label = -1 if "cat" in filename else 1
            labels.append(label)

            if max_per_class is not None:
                class_counts[class_type] += 1

    df = pd.DataFrame({"image": images, "label": labels})
    print(
        f"\nLoaded {len(images)} images: {class_counts['cat']} cats, {class_counts['dog']} dogs"
    )
    return df


def preprocess_image(img, target_size=(64, 64), to_gray=True):
    """Resize image to target size and flatten."""
    img = Image.fromarray(img)
    if to_gray:
        img = img.convert("L")
    img = img.resize(target_size)
    img_array = np.array(img).flatten() / 255.0
    return img_array


def solve_svm_nominal(x, y, lambda_reg):
    """Solve nominal SVM optimization problem."""
    n, d = x.shape
    w = cp.Variable(d)
    b = cp.Variable()
    zeta = cp.Variable(n)

    ## Constraints
    constraints = [zeta >= 0, cp.multiply(y, x @ w - b) >= 1 - zeta]

    ## Objective: minimize regularization + hinge loss
    objective = cp.Minimize(lambda_reg * cp.sum_squares(w) + (1.0 / n) * cp.sum(zeta))

    ## Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, **solver_opts)

    print(f"Solver status: {prob.status}")
    print(f"Objective value: {prob.value:.6f}")

    return w.value, b.value


def evaluate_accuracy(x, y, w, b):
    """Calculate accuracy and confusion matrix."""
    scores = x @ w - b
    y_pred = np.where(scores >= 0, 1, -1)
    accuracy = accuracy_score(y, y_pred)
    confusion_mat = confusion_matrix(y, y_pred)
    return accuracy, confusion_mat


def main():
    parser = argparse.ArgumentParser(
        description="Simple SVM for cats vs dogs classification"
    )
    parser.add_argument(
        "--dimension", type=int, default=16, help="Image dimension (default: 16)"
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=300,
        help="Max images per class (default: 300)",
    )
    parser.add_argument(
        "--lambda-reg",
        type=float,
        default=0.1,
        help="Regularization parameter (default: 0.1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test set ratio (default: 0.5)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SIMPLE SVM - CATS VS DOGS CLASSIFICATION")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Image dimension: {args.dimension}x{args.dimension}")
    print(f"  Max images per class: {args.max_per_class}")
    print(f"  Regularization (lambda): {args.lambda_reg}")
    print(f"  Random seed: {args.seed}")
    print(f"  Test size ratio: {args.test_size}")
    print("=" * 70)

    ## Load images
    df = load_images(
        dimension=args.dimension, load_in_gray=True, max_per_class=args.max_per_class
    )
    x = np.array(df["image"].tolist())
    y = df["label"].values

    ## Set random seed
    np.random.seed(args.seed)

    ## Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed
    )

    print(f"\nData split:")
    print(
        f"  Training: {len(y_train)} samples ({(y_train==-1).sum()} cats, {(y_train==1).sum()} dogs)"
    )
    print(
        f"  Testing:  {len(y_test)} samples ({(y_test==-1).sum()} cats, {(y_test==1).sum()} dogs)"
    )

    ## Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print(f"\nData standardized:")
    print(f"  Train - mean: {x_train.mean():.6f}, std: {x_train.std():.6f}")
    print(f"  Test  - mean: {x_test.mean():.6f}, std: {x_test.std():.6f}")

    ## Train SVM
    print("\n" + "=" * 70)
    print("TRAINING SVM...")
    print("=" * 70)
    w, b = solve_svm_nominal(x_train, y_train, lambda_reg=args.lambda_reg)

    ## Evaluate on training set
    train_acc, train_conf = evaluate_accuracy(x_train, y_train, w, b)

    ## Evaluate on test set
    test_acc, test_conf = evaluate_accuracy(x_test, y_test, w, b)

    ## Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTraining Set Performance:")
    print(f"  Accuracy: {train_acc*100:.2f}%")
    print(f"  Confusion Matrix:")
    print(f"    {train_conf}")

    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {test_acc*100:.2f}%")
    print(f"  Confusion Matrix:")
    print(f"    {test_conf}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    ## Check overfitting
    overfitting_gap = (train_acc - test_acc) * 100
    print(f"  Train-Test Gap: {overfitting_gap:.2f}%")

    if overfitting_gap > 10:
        print("  ⚠️  Warning: Significant overfitting detected!")
        print("      Consider: Increase lambda_reg, reduce dimension, or add more data")
    elif overfitting_gap < 0:
        print("  ⚠️  Unusual: Test accuracy > Train accuracy")
    else:
        print("  ✓ Good generalization")

    ## Decision scores analysis
    scores_train = x_train @ w - b
    scores_test = x_test @ w - b

    print(f"\nDecision scores (training):")
    print(
        f"  Cats (label=-1): mean={scores_train[y_train==-1].mean():.4f}, std={scores_train[y_train==-1].std():.4f}"
    )
    print(
        f"  Dogs (label=1):  mean={scores_train[y_train==1].mean():.4f}, std={scores_train[y_train==1].std():.4f}"
    )

    print(f"\nDecision scores (test):")
    print(
        f"  Cats (label=-1): mean={scores_test[y_test==-1].mean():.4f}, std={scores_test[y_test==-1].std():.4f}"
    )
    print(
        f"  Dogs (label=1):  mean={scores_test[y_test==1].mean():.4f}, std={scores_test[y_test==1].std():.4f}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
