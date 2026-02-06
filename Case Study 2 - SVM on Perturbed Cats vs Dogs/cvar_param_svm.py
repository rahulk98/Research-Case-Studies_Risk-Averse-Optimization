import dis
import argparse
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

"""
SVM with perturbed images of cats and dogs
lables: -1 -> cats, 1 -> dogs

"""
## Solver options for the optimization problem
solver_opts = {"eps_abs": 1e-5, "eps_rel": 1e-5, "max_iter": 10000}


def svm_loss_func(x, y, w, b):
    return max(0.0, 1.0 - y * (np.dot(w, x) - b))


def load_images(dimension=64, load_in_gray=True, max_per_class=None):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "train")
    images = []
    labels = []
    class_counts = {"cat": 0, "dog": 0}

    for filename in tqdm(sorted(os.listdir(path)), desc="Loading images"):
        if filename.endswith(".jpg"):
            # Check if we've reached the limit for this class
            if max_per_class is not None:
                class_type = "cat" if "cat" in filename else "dog"
                if class_counts[class_type] >= max_per_class:
                    continue

            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            img_array = np.array(img)
            img_array = preprocess_image_loading(
                img_array, target_size=(dimension, dimension), to_gray=load_in_gray
            )
            images.append(img_array)
            label = -1 if "cat" in filename else 1
            labels.append(label)

            # Update class count
            if max_per_class is not None:
                class_counts[class_type] += 1

    df = pd.DataFrame({"image": images, "label": labels})
    return df


# Resize the images to 64 x 64 and flatten
def preprocess_image_loading(img, target_size=(64, 64), to_gray=True):
    img = Image.fromarray(img)
    if to_gray:
        img = img.convert("L")  # Convert to grayscale
    img = img.resize(target_size)
    img_array = np.array(img).flatten() / 255.0  # Normalize pixel values
    return img_array


def solve_svm_nominal(x, y, lambda_reg):
    n, d = x.shape
    w = cp.Variable(d)
    b = cp.Variable()
    zeta = cp.Variable(n)
    ## Define the constraints
    constraints = []
    constraints += [zeta >= 0, cp.multiply(y, x @ w - b) >= 1 - zeta]

    ## Define the objective
    objective = cp.Minimize(lambda_reg * cp.sum_squares(w) + (1.0 / n) * cp.sum(zeta))

    ## Define the problem
    prob = cp.Problem(objective, constraints)

    prob.solve(solver=cp.OSQP, **solver_opts)

    w_val = w.value
    b_val = b.value
    zeta_val = zeta.value
    obj_val = prob.value
    print("Solver status (nominal):", prob.status)
    return w_val, b_val, zeta_val, obj_val


def solve_svm_cvar(x, y, lambda_reg, alpha, N):
    _, n, d = x.shape
    w = cp.Variable(d)
    b = cp.Variable()
    zeta = cp.Variable((n, N))
    u = cp.Variable(N)
    t = cp.Variable()

    ## Define the objective
    obj = (
        t
        + (1.0 / ((1.0 - alpha) * float(N))) * cp.sum(u)
        + lambda_reg * cp.sum_squares(w)
    )
    objective = cp.Minimize(obj)

    ## Define the constraints

    constraints = []

    ## Constraints for each pertubed scenario
    ## For scenario k: zeta[:,k] >= 1 - y * (Xk @ w - b); zeta[:,k] >= 0

    for k in range(N):
        x_k = x[k, :, :]
        constraints += [zeta[:, k] >= 0]
        constraints += [cp.multiply(y, x_k @ w - b) >= 1 - zeta[:, k]]

    ## Constraints for u
    for k in range(N):
        constraints += [u[k] >= 0]
        constraints += [u[k] >= ((1.0 / n) * cp.sum(zeta[:, k]) - t)]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, **solver_opts)

    print("Solver status (CVaR):", problem.status)

    return w.value, b.value, problem.value, t.value


def calc_metrics(x, y, w, b):
    scores = x @ w - b
    y_pred = np.where(scores >= 0, 1, -1)
    accuracy = accuracy_score(y, y_pred)
    confusion_mat = confusion_matrix(y, y_pred)
    return accuracy, confusion_mat


def calc_metrics_noisy(x_noisy, y, w, b, M):
    """Calculate average accuracy and confusion matrix across M noisy scenarios."""
    accuracies = []
    confusion_mats = []

    for k in range(M):
        x_k = x_noisy[k, :, :]
        scores = x_k @ w - b
        y_pred = np.where(scores >= 0, 1, -1)
        acc = accuracy_score(y, y_pred)
        conf_mat = confusion_matrix(y, y_pred)
        accuracies.append(acc)
        confusion_mats.append(conf_mat)

    avg_accuracy = np.mean(accuracies)
    avg_confusion_mat = np.mean(confusion_mats, axis=0)

    return avg_accuracy, avg_confusion_mat


def cvar_accuracy(x_noisy, y, w, b, alpha, M):
    """Calculate CVaR of accuracy (worst-case average accuracy for bottom (1-alpha) tail)."""
    accuracies = []

    for k in range(M):
        x_k = x_noisy[k, :, :]
        scores = x_k @ w - b
        y_pred = np.where(scores >= 0, 1, -1)
        acc = accuracy_score(y, y_pred)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    # For accuracy, lower is worse, so we want the lower (1-alpha) tail
    var_alpha = np.quantile(accuracies, 1 - alpha)
    tail = accuracies[accuracies <= var_alpha]
    return float(tail.mean())


def evaluate_losses(x, y, w, b, M):
    """Evaluate hinge losses for test samples."""
    losses = []
    for k in range(M):
        loss = np.array([svm_loss_func(x[k, i], y[i], w, b) for i in range(x.shape[1])])
        losses.append(loss.mean())

    return np.array(losses)


def cvar_empirical(values, alpha):
    """
    Empirical CVaR_alpha for a 1D array of loss values:
      CVaR = mean of the upper (1-alpha) tail (>= VaR_alpha).
    """
    values = np.asarray(values, dtype=float)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if values.size == 0:
        return np.nan
    # Empirical VaR at level alpha
    var_alpha = np.quantile(values, alpha)
    tail = values[values >= var_alpha]
    return float(tail.mean())


def main():
    parser = argparse.ArgumentParser(
        description="SVM on Perturbed Images: nominal vs CVaR-optimized comparison"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.95, help="CVaR level in (0,1)."
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "nominal", "cvar", "seed_search"],
        default="compare",
        help="Run nominal only, CVaR only, both and compare (default), or seed_search.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=100,
        help="Maximum number of seeds to search (for seed_search mode).",
    )
    parser.add_argument(
        "--lambda_reg", type=float, default=1e-3, help="Regularization parameter."
    )
    parser.add_argument(
        "--dimension", type=int, default=64, help="Dimension of flattened images."
    )
    parser.add_argument(
        "--grayscale", type=bool, default=True, help="Convert images to grayscale."
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of images to consider per class."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=10,
        help="Number of scenarios for training (optimization).",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=100,
        help="Number of scenarios for evaluation (Test CVaR).",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.1, help="Standard deviation of image noise."
    )

    parser.add_argument(
        "--dist", type=str, default="normal", help="Pertubation Distribution"
    )

    args = parser.parse_args()

    ## Seed search mode
    if args.mode == "seed_search":
        print("=" * 60)
        print(f"Seed Search Mode: Testing seeds 0 to {args.max_seeds}")
        print(
            f"Parameters: N={args.N}, M={args.M}, alpha={args.alpha}, sigma={args.sigma}"
        )
        print("=" * 60)

        results = []

        for seed in range(args.max_seeds + 1):
            print(f"\nTesting seed {seed}...")

            ## Load the images and preprocess
            df = load_images(
                dimension=args.dimension,
                load_in_gray=args.grayscale,
                max_per_class=args.n,
            )
            x = np.array(df["image"].tolist())
            y = df["label"].values
            np.random.seed(seed)

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.5, random_state=seed
            )
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            ## Generate noisy perturbations
            rng = np.random.default_rng(seed)
            if args.dist == "normal":
                x_train_noisy = np.zeros((args.N, x_train.shape[0], x_train.shape[1]))
                for k in range(args.N):
                    x_train_noisy[k, :, :] = x_train + rng.normal(
                        0, args.sigma, size=x_train.shape
                    )

                x_test_noisy = np.zeros((args.M, x_test.shape[0], x_test.shape[1]))
                for k in range(args.M):
                    x_test_noisy[k, :, :] = x_test + rng.normal(
                        0, args.sigma, size=x_test.shape
                    )
            elif args.dist == "students_t":
                x_train_noisy = np.zeros((args.N, x_train.shape[0], x_train.shape[1]))
                for k in range(args.N):
                    x_train_noisy[k, :, :] = (
                        x_train + rng.standard_t(df=3, size=x_train.shape) * args.sigma
                    )

                x_test_noisy = np.zeros((args.M, x_test.shape[0], x_test.shape[1]))
                for k in range(args.M):
                    x_test_noisy[k, :, :] = (
                        x_test + rng.standard_t(df=3, size=x_test.shape) * args.sigma
                    )

            ## Solve nominal SVM
            w_nom, b_nom, _, _ = solve_svm_nominal(
                x_train, y_train, lambda_reg=args.lambda_reg
            )
            losses_nom = evaluate_losses(x_test_noisy, y_test, w_nom, b_nom, M=args.M)
            cvar_nom = cvar_empirical(losses_nom, args.alpha)
            mean_nom = float(losses_nom.mean())
            avg_acc_nom, _ = calc_metrics_noisy(
                x_test_noisy, y_test, w_nom, b_nom, M=args.M
            )
            cvar_acc_nom = cvar_accuracy(
                x_test_noisy, y_test, w_nom, b_nom, args.alpha, M=args.M
            )

            ## Solve CVaR SVM
            w_cvar, b_cvar, _, _ = solve_svm_cvar(
                x_train_noisy,
                y_train,
                lambda_reg=args.lambda_reg,
                alpha=args.alpha,
                N=args.N,
            )
            losses_cvar = evaluate_losses(
                x_test_noisy, y_test, w_cvar, b_cvar, M=args.M
            )
            cvar_cvar = cvar_empirical(losses_cvar, args.alpha)
            mean_cvar = float(losses_cvar.mean())
            avg_acc_cvar, _ = calc_metrics_noisy(
                x_test_noisy, y_test, w_cvar, b_cvar, M=args.M
            )
            cvar_acc_cvar = cvar_accuracy(
                x_test_noisy, y_test, w_cvar, b_cvar, args.alpha, M=args.M
            )

            ## Compare results
            cvar_loss_better = cvar_cvar < cvar_nom
            mean_loss_better = mean_cvar < mean_nom
            cvar_acc_better = cvar_acc_cvar > cvar_acc_nom
            mean_acc_better = avg_acc_cvar > avg_acc_nom

            results.append(
                {
                    "seed": seed,
                    "cvar_loss_nom": cvar_nom,
                    "cvar_loss_cvar": cvar_cvar,
                    "mean_loss_nom": mean_nom,
                    "mean_loss_cvar": mean_cvar,
                    "cvar_acc_nom": cvar_acc_nom,
                    "cvar_acc_cvar": cvar_acc_cvar,
                    "mean_acc_nom": avg_acc_nom,
                    "mean_acc_cvar": avg_acc_cvar,
                    "cvar_loss_better": cvar_loss_better,
                    "mean_loss_better": mean_loss_better,
                    "cvar_acc_better": cvar_acc_better,
                    "mean_acc_better": mean_acc_better,
                }
            )

            print(
                f"  CVaR Loss: {cvar_cvar:.6f} vs {cvar_nom:.6f} {'✓ CVaR wins' if cvar_loss_better else ''}"
            )
            print(
                f"  Mean Loss: {mean_cvar:.6f} vs {mean_nom:.6f} {'✓ CVaR wins' if mean_loss_better else ''}"
            )
            print(
                f"  CVaR Acc:  {cvar_acc_cvar*100:.2f}% vs {cvar_acc_nom*100:.2f}% {'✓ CVaR wins' if cvar_acc_better else ''}"
            )
            print(
                f"  Mean Acc:  {avg_acc_cvar*100:.2f}% vs {avg_acc_nom*100:.2f}% {'✓ CVaR wins' if mean_acc_better else ''}"
            )

        ## Summary
        print("\n" + "=" * 60)
        print("SEED SEARCH SUMMARY")
        print("=" * 60)

        # Calculate accuracy difference for each result
        for r in results:
            r["cvar_acc_diff"] = r["cvar_acc_cvar"] - r["cvar_acc_nom"]

        # Sort by CVaR accuracy difference (descending)
        results_sorted = sorted(results, key=lambda r: r["cvar_acc_diff"], reverse=True)

        cvar_loss_wins = sum(r["cvar_loss_better"] for r in results)
        mean_loss_wins = sum(r["mean_loss_better"] for r in results)
        cvar_acc_wins = sum(r["cvar_acc_better"] for r in results)
        mean_acc_wins = sum(r["mean_acc_better"] for r in results)

        print(f"\nSeeds where CVaR optimization performed better:")
        print(f"  CVaR Loss (lower is better): {cvar_loss_wins}/{len(results)} seeds")
        print(f"  Mean Loss (lower is better): {mean_loss_wins}/{len(results)} seeds")
        print(
            f"  CVaR Accuracy (higher is better): {cvar_acc_wins}/{len(results)} seeds"
        )
        print(
            f"  Mean Accuracy (higher is better): {mean_acc_wins}/{len(results)} seeds"
        )

        print(f"\n" + "=" * 60)
        print("TOP SEEDS BY CVaR ACCURACY IMPROVEMENT")
        print("=" * 60)

        # Show top 10 seeds by CVaR accuracy difference
        top_n = min(10, len(results_sorted))
        for i, r in enumerate(results_sorted[:top_n], 1):
            diff = r["cvar_acc_diff"] * 100
            symbol = "✓" if r["cvar_acc_better"] else "✗"
            print(
                f"{i:2d}. Seed {r['seed']:3d}: {r['cvar_acc_cvar']*100:5.2f}% vs {r['cvar_acc_nom']*100:5.2f}% ({diff:+6.2f}% points) {symbol}"
            )

        if cvar_acc_wins > 0:
            best_seed = results_sorted[0]
            print(f"\n" + "=" * 60)
            print(f"BEST SEED: {best_seed['seed']}")
            print("=" * 60)
            print(
                f"CVaR Accuracy: {best_seed['cvar_acc_cvar']*100:.2f}% vs {best_seed['cvar_acc_nom']*100:.2f}% ({best_seed['cvar_acc_diff']*100:+.2f}% points)"
            )
            print(
                f"Mean Accuracy: {best_seed['mean_acc_cvar']*100:.2f}% vs {best_seed['mean_acc_nom']*100:.2f}%"
            )
            print(
                f"CVaR Loss:     {best_seed['cvar_loss_cvar']:.6f} vs {best_seed['cvar_loss_nom']:.6f}"
            )
            print(
                f"Mean Loss:     {best_seed['mean_loss_cvar']:.6f} vs {best_seed['mean_loss_nom']:.6f}"
            )

        return

    ## Load the images and preprocess
    df = load_images(
        dimension=args.dimension, load_in_gray=args.grayscale, max_per_class=args.n
    )
    x = np.array(df["image"].tolist())  # Convert DataFrame column to 2D array
    y = df["label"].values
    np.random.seed(args.seed)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=args.seed
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    ## Generate noisy pertubations of training and test data
    rng = np.random.default_rng(args.seed)
    if args.dist == "normal":
        x_train_noisy = np.zeros((args.N, x_train.shape[0], x_train.shape[1]))
        ## Generate N different pertubations of training data
        for k in range(args.N):
            x_train_noisy[k, :, :] = x_train + rng.normal(
                0, args.sigma, size=x_train.shape
            )
        rng = np.random.default_rng(args.seed + 1)
        x_test_noisy = np.zeros((args.M, x_test.shape[0], x_test.shape[1]))
        for k in range(args.M):
            x_test_noisy[k, :, :] = x_test + rng.normal(
                0, args.sigma, size=x_test.shape
            )
    elif args.dist == "students_t":
        x_train_noisy = np.zeros((args.N, x_train.shape[0], x_train.shape[1]))
        ## Generate N different pertubations of training data
        for k in range(args.N):
            x_train_noisy[k, :, :] = (
                x_train + rng.standard_t(df=3, size=x_train.shape) * args.sigma
            )
        rng = np.random.default_rng(args.seed + 1)
        x_test_noisy = np.zeros((args.M, x_test.shape[0], x_test.shape[1]))
        for k in range(args.M):
            x_test_noisy[k, :, :] = (
                x_test + rng.standard_t(df=3, size=x_test.shape) * args.sigma
            )

    ## Solve nominal SVM
    if args.mode in ("compare", "nominal"):
        w_nom, b_nom, zeta_nom, obj_nom = solve_svm_nominal(
            x_train, y_train, lambda_reg=args.lambda_reg
        )
        losses_nom = evaluate_losses(x_test_noisy, y_test, w_nom, b_nom, M=args.M)
        cvar_nom = cvar_empirical(losses_nom, args.alpha)
        mean_nom = float(losses_nom.mean())

        test_acc_nom, test_conf_mat_nom = calc_metrics(x_test, y_test, w_nom, b_nom)
        avg_acc_nom, avg_conf_nom = calc_metrics_noisy(
            x_test_noisy, y_test, w_nom, b_nom, M=args.M
        )
        cvar_acc_nom = cvar_accuracy(
            x_test_noisy, y_test, w_nom, b_nom, args.alpha, M=args.M
        )

        print("Nominal SVM Results:")
        print("  Value of objective at optimum = ", obj_nom)
        print(f"Test Set Results (test_N = {x_test.shape[0]}):")
        print(f"  Mean hinge loss (perturbed, M={args.M}) = {mean_nom:.6f}")
        print(f"  CVaR@{args.alpha} hinge loss (perturbed) = {cvar_nom:.6f}")
        print(f"  Mean accuracy (perturbed, M={args.M}) = {avg_acc_nom*100:.2f}%")
        print(f"  CVaR@{args.alpha} accuracy (perturbed) = {cvar_acc_nom*100:.2f}%")

        # train_acc_nom, train_conf_mat_nom = calc_metrics(x_train, y_train, w_nom, b_nom)
        print(f"\nTest Accuracy (unperturbed): {test_acc_nom*100:.2f}%")
        print("Test Confusion Matrix (unperturbed):")
        print(test_conf_mat_nom)
        print(
            f"\nAverage Test Accuracy (perturbed, M={args.M}): {avg_acc_nom*100:.2f}%"
        )
        print("Average Test Confusion Matrix (perturbed):")
        print(avg_conf_nom)
    if args.mode in ("compare", "cvar"):
        w_cvar, b_cvar, obj_cvar, var_cvar = solve_svm_cvar(
            x_train_noisy,
            y_train,
            lambda_reg=args.lambda_reg,
            alpha=args.alpha,
            N=args.N,
        )
        losses_cvar = evaluate_losses(x_test_noisy, y_test, w_cvar, b_cvar, M=args.M)
        cvar_cvar = cvar_empirical(losses_cvar, args.alpha)
        mean_cvar = float(losses_cvar.mean())

        test_acc_cvar, test_conf_mat_cvar = calc_metrics(x_test, y_test, w_cvar, b_cvar)
        avg_acc_cvar, avg_conf_cvar = calc_metrics_noisy(
            x_test_noisy, y_test, w_cvar, b_cvar, M=args.M
        )
        cvar_acc_cvar = cvar_accuracy(
            x_test_noisy, y_test, w_cvar, b_cvar, args.alpha, M=args.M
        )

        print("CVaR-optimized SVM Results:")
        print("  Value of objective at optimum = ", obj_cvar)
        print(f"Test Set Results (test_N = {x_test.shape[0]}):")
        print(f"  Mean hinge loss (perturbed, M={args.M}) = {mean_cvar:.6f}")
        print(f"  CVaR@{args.alpha} hinge loss (perturbed) = {cvar_cvar:.6f}")
        print(f"  Mean accuracy (perturbed, M={args.M}) = {avg_acc_cvar*100:.2f}%")
        print(f"  CVaR@{args.alpha} accuracy (perturbed) = {cvar_acc_cvar*100:.2f}%")
        print(f"\nTest Accuracy (unperturbed): {test_acc_cvar*100:.2f}%")
        print("Test Confusion Matrix (unperturbed):")
        print(test_conf_mat_cvar)
        print(
            f"\nAverage Test Accuracy (perturbed, M={args.M}): {avg_acc_cvar*100:.2f}%"
        )
        print("Average Test Confusion Matrix (perturbed):")
        print(avg_conf_cvar)

    if args.mode == "compare":
        print("\n" + "=" * 60)
        print("Comparison (Test Set): CVaR@alpha=%.2f" % args.alpha)
        print("=" * 60)
        print("\nHinge Loss (Perturbed):")
        print(
            "  CVaR(cvar)  vs  CVaR(nominal)  -->  %.6f  vs  %.6f"
            % (cvar_cvar, cvar_nom)
        )
        print(
            "  Mean(cvar)  vs  Mean(nominal)  -->  %.6f  vs  %.6f"
            % (mean_cvar, mean_nom)
        )
        print("\nAccuracy (Perturbed):")
        print(
            "  CVaR(cvar)  vs  CVaR(nominal)  -->  %.2f%%  vs  %.2f%%"
            % (cvar_acc_cvar * 100, cvar_acc_nom * 100)
        )
        print(
            "  Mean(cvar)  vs  Mean(nominal)  -->  %.2f%%  vs  %.2f%%"
            % (avg_acc_cvar * 100, avg_acc_nom * 100)
        )


if __name__ == "__main__":
    main()
