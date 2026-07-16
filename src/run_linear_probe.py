from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.analysis.classification_plots import (
    save_confusion_matrix_plot,
)
from src.analysis.linear_probe import (
    fit_linear_probe,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen histology embeddings "
            "with a linear classification probe."
        )
    )

    parser.add_argument(
        "--train-embeddings",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validation-embeddings",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--test-embeddings",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[
            0.001,
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--feature-prefix",
        type=str,
        default="latent_",
        help=(
            "Prefix of classifier feature columns, "
            "for example latent_ or color_."
        ),
    )
    
    parser.add_argument(
        "--pca-components",
        type=int,
        default=None,
        help=(
            "Optional PCA dimensionality applied, "
            "inside the training pipeline."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    train_embeddings = pd.read_csv(
        arguments.train_embeddings
    )

    validation_embeddings = pd.read_csv(
        arguments.validation_embeddings
    )

    test_embeddings = pd.read_csv(
        arguments.test_embeddings
    )

    result = fit_linear_probe(
        train_embeddings=train_embeddings,
        validation_embeddings=(
            validation_embeddings
        ),
        test_embeddings=test_embeddings,
        c_values=arguments.c_values,
        seed=arguments.seed,
        feature_prefix=arguments.feature_prefix,
        pca_components=arguments.pca_components,
    )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.selection_results.to_csv(
        arguments.output_dir
        / "model_selection.csv",
        index=False,
    )

    result.class_table.to_csv(
        arguments.output_dir
        / "class_mapping.csv",
        index=False,
    )

    result.validation.predictions.to_csv(
        arguments.output_dir
        / "validation_predictions.csv",
        index=False,
    )

    result.test.predictions.to_csv(
        arguments.output_dir
        / "test_predictions.csv",
        index=False,
    )

    result.validation.classification_report.to_csv(
        arguments.output_dir
        / "validation_classification_report.csv",
        index=False,
    )

    result.test.classification_report.to_csv(
        arguments.output_dir
        / "test_classification_report.csv",
        index=False,
    )

    class_codes = (
        result.class_table[
            "class_code"
        ].astype(str).tolist()
    )

    validation_confusion = pd.DataFrame(
        result.validation.confusion_matrix,
        index=class_codes,
        columns=class_codes,
    )

    validation_confusion.to_csv(
        arguments.output_dir
        / "validation_confusion_matrix.csv"
    )

    test_confusion = pd.DataFrame(
        result.test.confusion_matrix,
        index=class_codes,
        columns=class_codes,
    )

    test_confusion.to_csv(
        arguments.output_dir
        / "test_confusion_matrix.csv"
    )

    save_confusion_matrix_plot(
        matrix=(
            result.validation
            .normalized_confusion_matrix
        ),
        class_codes=class_codes,
        output_path=(
            arguments.output_dir
            / "validation_confusion_matrix.png"
        ),
        title=(
            "Validation confusion matrix "
            "(row-normalized)"
        ),
        normalized=True,
    )

    save_confusion_matrix_plot(
        matrix=(
            result.test
            .normalized_confusion_matrix
        ),
        class_codes=class_codes,
        output_path=(
            arguments.output_dir
            / "external_test_confusion_matrix.png"
        ),
        title=(
            "External test confusion matrix "
            "(row-normalized)"
        ),
        normalized=True,
    )

    joblib.dump(
        result.final_model,
        arguments.output_dir
        / "linear_probe.joblib",
    )

    metrics = {
        "best_c": result.best_c,
        "feature_prefix": result.feature_prefix,

        # Размерность исходных признаков до PCA.
        "input_feature_dimensions": len(
            result.feature_columns
        ),

        # None, если PCA не используется.
        "pca_components": (
            result.pca_components
        ),

        # Размерность признаков, поступающих
        # непосредственно в LogisticRegression.
        "output_feature_dimensions": (
            result.pca_components
            if result.pca_components is not None
            else len(result.feature_columns)
        ),

        "train_images": int(
            len(train_embeddings)
        ),
        "validation_images": int(
            len(validation_embeddings)
        ),
        "external_test_images": int(
            len(test_embeddings)
        ),

        # Модель обучена только на train.
        "validation_train_only": (
            result.validation.metrics
        ),

        # Та же самая train-only модель,
        # применённая к external test.
        "external_test_train_only": (
            result.test_before_refit.metrics
        ),

        # Финальная модель после переобучения
        # на train + validation.
        "external_test_after_refit": (
            result.test.metrics
        ),

        # Сохраняем старые ключи для совместимости
        # с уже написанными скриптами.
        "validation": (
            result.validation.metrics
        ),
        "external_test": (
            result.test.metrics
        ),

        # Честный domain gap:
        # validation и external test оценены
        # одной и той же train-only моделью.
        "balanced_accuracy_gap_train_only": float(
            result.validation.metrics[
                "balanced_accuracy"
            ]
            - result.test_before_refit.metrics[
                "balanced_accuracy"
            ]
        ),

        "macro_f1_gap_train_only": float(
            result.validation.metrics[
                "macro_f1"
            ]
            - result.test_before_refit.metrics[
                "macro_f1"
            ]
        ),

        # Насколько переобучение на train + validation
        # изменило результат external test.
        "external_refit_gain": float(
            result.test.metrics[
                "balanced_accuracy"
            ]
            - result.test_before_refit.metrics[
                "balanced_accuracy"
            ]
        ),

        "external_refit_macro_f1_gain": float(
            result.test.metrics[
                "macro_f1"
            ]
            - result.test_before_refit.metrics[
                "macro_f1"
            ]
        ),

        # Старые поля оставляем только для совместимости.
        # Они сравнивают validation train-only модели
        # с test после refit, поэтому для научной
        # интерпретации их использовать не следует.
        "balanced_accuracy_gap": float(
            result.validation.metrics[
                "balanced_accuracy"
            ]
            - result.test.metrics[
                "balanced_accuracy"
            ]
        ),

        "macro_f1_gap": float(
            result.validation.metrics[
                "macro_f1"
            ]
            - result.test.metrics[
                "macro_f1"
            ]
        ),

        # Совместимость со старыми версиями
        # compare_probe_baselines.py.
        "feature_dimensions": len(
            result.feature_columns
        ),

        "latent_dimensions": (
            len(result.feature_columns)
            if result.feature_prefix == "latent_"
            and result.pca_components is None
            else (
                result.pca_components
                if result.feature_prefix == "latent_"
                else None
            )
        ),
    }

    with (
        arguments.output_dir / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("=" * 68)
    print("CRC LATENT LINEAR PROBE")
    print("=" * 68)

    print(f"Best C: {result.best_c:g}")

    print(
        f"Feature prefix: {result.feature_prefix!r}"
    )
    
    print(
        f"Feature dimensions: "
        f"{len(result.feature_columns)}"
    )
    print(
        f"Input dimensions:     "
        f"{len(result.feature_columns)}"
    )

    if result.pca_components is None:
        print("PCA components:       disabled")
    else:
        print(
            f"PCA components:       "
            f"{result.pca_components}"
        )
    print()
    print("Validation")
    for metric_name, metric_value in (
        result.validation.metrics.items()
    ):
        print(
            f"  {metric_name:20s} "
            f"{metric_value:.4f}"
        )
    print()
    print()
    print("External test — train-only model")

    for metric_name, metric_value in (
        result.test_before_refit.metrics.items()
    ):
        print(
            f"  {metric_name:24s} "
            f"{metric_value:.4f}"
        )
    print("External test — after train+validation refit")
    for metric_name, metric_value in (
        result.test.metrics.items()
    ):
        print(
            f"  {metric_name:24s} "
            f"{metric_value:.4f}"
        )

    print()
    print(
        "Balanced accuracy gap: "
        f"{metrics['balanced_accuracy_gap']:.4f}"
    )
    print()
    print(
        "Train-only balanced accuracy gap: "
        f"{metrics['balanced_accuracy_gap_train_only']:.4f}"
    )

    print(
        "External refit gain: "
        f"{metrics['external_refit_gain']:.4f}"
    )
    print(
        f"Results: "
        f"{arguments.output_dir.resolve()}"
    )

    print("=" * 68)


if __name__ == "__main__":
    main()