"""
PreProcessor module for MLArena package.
"""

import re
from typing import Any, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder


class PreProcessor(BaseEstimator, TransformerMixin):
    """
    Custom transformer for data preprocessing with scikit-learn compatibility.

    - preprocessing strategy support
        - feature analysis and filter feature selection recommendations
        - encoding recommendations of categorical features based on cardinality and rare category
        - visualization to compare smoothing parameters for target encoding
    - preprocessing execution
        - scaling of numeric features
        - target encoding of user specified target encode columns
        - onehot encoding of the rest of the categorical features
        - Handles missing values via user specified imputation strategy for numeric and categorical features
    - Compatible with scikit-learn pipelines and models

    Parameters
    ----------
    num_impute_strategy : str, default="median"
        Strategy for numeric missing values.
    cat_impute_strategy : str, default="most_frequent"
        Strategy for categorical missing values.
    target_encode_cols : List[str], optional
        Columns to apply mean encoding.
    target_encode_smooth : Union[str, float], default="auto"
        Smoothing parameter for target encoding.
    drop : str, default="if_binary"
        Strategy for dropping categories in OneHotEncoder.
        Options: 'if_binary', 'first', None.
    sanitize_feature_names : bool, default=True
        Whether to sanitize feature names by replacing special characters.

    Attributes
    ----------
    num_impute_strategy : str
        Numeric imputation strategy.
    cat_impute_strategy : str
        Categorical imputation strategy.
    num_transformer : Pipeline
        Numeric preprocessing pipeline.
    cat_transformer : Pipeline
        Categorical preprocessing pipeline.
    target_transformer : Pipeline
        Target encoding preprocessing pipeline.
    transformed_cat_cols : List[str]
        One-hot encoded column names.
    num_features : List[str]
        Numeric feature names.
    cat_features : List[str]
        Categorical feature names.
    target_encode_cols : List[str]
        Columns for target encoding.
    target_encode_smooth : Union[str, float]
        Smoothing parameter for target encoding.
    drop : str
        Strategy for dropping categories in OneHotEncoder.
    sanitize_feature_names : bool
        Whether to sanitize feature names by replacing special characters.

    """

    def __init__(
        self,
        num_impute_strategy="median",
        cat_impute_strategy="most_frequent",
        target_encode_cols=None,
        target_encode_smooth="auto",
        drop="if_binary",  # choose "first" for linear models and "if_binary" for tree models
        sanitize_feature_names=True,
    ):
        """
        Initialize the transformer.

        Parameters
        ----------
        num_impute_strategy : str, default="median"
            Strategy for numeric missing values.
        cat_impute_strategy : str, default="most_frequent"
            Strategy for categorical missing values.
        target_encode_cols : List[str], optional
            Columns to apply mean encoding.
        target_encode_smooth : Union[str, float], default="auto"
            Smoothing parameter for target encoding.
        drop : str, default="if_binary"
            Strategy for dropping categories in OneHotEncoder.
            Options: 'if_binary', 'first', None.
        sanitize_feature_names : bool, default=True
            Whether to sanitize feature names by replacing special characters.
        """
        self.num_impute_strategy = num_impute_strategy
        self.cat_impute_strategy = cat_impute_strategy
        self.target_encode_cols = target_encode_cols
        self.target_encode_smooth = target_encode_smooth
        self.drop = drop
        self.sanitize_feature_names = sanitize_feature_names

    def fit_transform(self, X, y=None):
        """
        Fit transformer on input data and transform it.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable, required when target_encode_cols is specified.

        Returns
        -------
        pd.DataFrame
            Transformed data.

        Raises
        ------
        ValueError
            If target variable y is not provided when target_encode_cols is specified,
            or if specified columns are not found in input data.
        """
        if self.target_encode_cols and y is None:
            raise ValueError(
                "Target variable y is required when target_encode_cols is specified"
            )

        if self.target_encode_cols:
            missing_cols = [
                col for col in self.target_encode_cols if col not in X.columns
            ]
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in input data")

        self.num_features = X.select_dtypes(include=np.number).columns.tolist()
        self.cat_features = [
            col
            for col in X.select_dtypes(exclude=np.number).columns
            if col not in (self.target_encode_cols or [])
        ]

        transformed_dfs = []

        # Handle target encoding features
        if self.target_encode_cols:
            # self.target_encode_cols = [f for f in self.target_encode_cols if f in X.columns] # if we want target coding to continue working even if some cols are missing
            self.target_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.cat_impute_strategy)),
                    (
                        "target_encoder",
                        TargetEncoder(smooth=self.target_encode_smooth, cv=5),
                    ),
                ]
            )
            target_encoded = self.target_transformer.fit_transform(
                X[self.target_encode_cols], y
            )
            transformed_dfs.append(
                pd.DataFrame(
                    target_encoded, columns=self.target_encode_cols, index=X.index
                )
            )

        # Handle numeric features
        if self.num_features:
            self.num_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.num_impute_strategy)),
                    ("scaler", StandardScaler()),
                ]
            )
            num_transformed = self.num_transformer.fit_transform(X[self.num_features])
            transformed_dfs.append(
                pd.DataFrame(num_transformed, columns=self.num_features, index=X.index)
            )

        # Handle categorical features
        if self.cat_features:
            self.cat_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=self.cat_impute_strategy)),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore", drop=self.drop, sparse_output=False
                        ),
                    ),
                ]
            )

            cat_transformed = self.cat_transformer.fit_transform(X[self.cat_features])
            transformed_dfs.append(
                pd.DataFrame(
                    cat_transformed,
                    columns=self.get_transformed_cat_cols(),
                    index=X.index,
                )
            )

        return pd.concat(transformed_dfs, axis=1)

    @staticmethod
    def _sanitize_feature_names(feature_names):
        """
        Sanitize feature names by replacing special characters.

        Parameters
        ----------
        feature_names : list
            List of feature names to sanitize.

        Returns
        -------
        list
            Sanitized feature names.
        """
        sanitized = []
        for name in feature_names:
            sanitized_name = str(name)

            # First: Replace characters with semantic meaning
            sanitized_name = (
                sanitized_name.replace("+", "_plus")
                .replace("%", "_pct")
                .replace("<", "_lt_")
                .replace(">", "_gt_")
                .replace("=", "_eq_")
            )

            # Second: Replace any remaining special characters with underscore
            # Keep only alphanumeric characters and underscores
            sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", sanitized_name)

            # Clean up: Remove multiple consecutive underscores
            while "__" in sanitized_name:
                sanitized_name = sanitized_name.replace("__", "_")

            # Remove leading/trailing underscores
            sanitized_name = sanitized_name.strip("_")

            sanitized.append(sanitized_name)
        return sanitized

    def get_transformed_cat_cols(self):
        """
        Get transformed categorical column names using sklearn's built-in method.

        Returns
        -------
        List[str]
            One-hot encoded column names.
        """
        if not hasattr(self, "cat_transformer"):
            return []

        # Get the encoder from the pipeline
        encoder = self.cat_transformer.named_steps["encoder"]

        feature_names = encoder.get_feature_names_out(self.cat_features)

        # Sanitize feature names if requested
        if self.sanitize_feature_names:
            return self._sanitize_feature_names(feature_names)
        else:
            return feature_names.tolist()

    def transform(self, X):
        """
        Transform input data.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Transformed data.

        Raises
        ------
        ValueError
            If transformer is not fitted.
        """
        if not hasattr(self, "num_features"):
            raise ValueError("Transformer not fitted. Call 'fit' before 'transform'.")

        transformed_parts = []  # Store transformed components

        if self.target_encode_cols and hasattr(self, "target_transformer"):
            target_encoded_data = self.target_transformer.transform(
                X[self.target_encode_cols]
            )
            target_encoded_df = pd.DataFrame(
                target_encoded_data, columns=self.target_encode_cols, index=X.index
            )
            transformed_parts.append(target_encoded_df)

        if self.num_features:
            transformed_num_data = self.num_transformer.transform(X[self.num_features])
            transformed_num_df = pd.DataFrame(
                transformed_num_data, columns=self.num_features, index=X.index
            )
            transformed_parts.append(transformed_num_df)

        if self.cat_features:
            transformed_cat_data = self.cat_transformer.transform(X[self.cat_features])
            self.transformed_cat_cols = self.get_transformed_cat_cols()
            transformed_cat_df = pd.DataFrame(
                transformed_cat_data, columns=self.transformed_cat_cols, index=X.index
            )
            transformed_parts.append(transformed_cat_df)

        # Concatenate all parts at once to avoid fragmentation
        X_transformed = pd.concat(transformed_parts, axis=1)

        X_transformed.index = X.index

        return X_transformed

    @staticmethod
    def plot_target_encoding_comparison(
        X_train, y_train, smooth_params, target_encode_col, figsize=(15, 6)
    ):
        """
        Plot target encoding comparison with different smoothing parameters.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Target variable.
        smooth_params : List[Union[str, float]]
            List of smoothing parameters to compare.
        target_encode_col : str
            Column name for target encoding.
        figsize : tuple, default=(15, 6)
            Figure size.

        Returns
        -------
        dict
            Dictionary containing the results for each smoothing parameter.
        """
        # Create subplots
        fig, axes = plt.subplots(1, len(smooth_params), figsize=figsize)
        if len(smooth_params) == 1:
            axes = [axes]

        results_dict = {}
        global_mean = y_train.mean()

        # Create plots for each smoothing parameter
        for ax, smooth in zip(axes, smooth_params):
            # Initialize and fit preprocessor
            preprocessor = PreProcessor(
                num_impute_strategy="median",
                cat_impute_strategy="most_frequent",
                target_encode_cols=[target_encode_col],
                target_encode_smooth=smooth,
            )
            X_processed = preprocessor.fit_transform(X_train, y_train)

            # Calculate statistics
            original_mean = y_train.groupby(
                X_train[target_encode_col], observed=False
            ).mean()
            encoded_mean = X_processed.groupby(
                X_train[target_encode_col], observed=False
            )[target_encode_col].mean()
            sample_size = X_train[target_encode_col].value_counts().sort_index()

            results = pd.concat(
                [
                    original_mean.rename("Original_Train_Mean"),
                    encoded_mean.rename("Encoded_Mean"),
                    sample_size.rename("Sample_Size"),
                ],
                axis=1,
            ).sort_index()

            results_dict[smooth] = results

            # Create bar plot
            results[["Original_Train_Mean", "Encoded_Mean"]].plot(
                kind="bar", width=0.8, ax=ax
            )
            ax.axhline(
                y=global_mean,
                color="r",
                linestyle="--",
                alpha=0.5,
                label=f"Global Mean ({global_mean:.3f})",
            )

            # Set title and labels
            smooth_label = "auto" if smooth == "auto" else str(smooth)
            ax.set_title(f"smooth={smooth_label}")
            ax.set_xlabel("Category")
            ax.set_ylabel("Mean Value")

            # Add sample sizes to x-labels
            ax.set_xticks(range(len(results)))
            ax.set_xticklabels(
                [f"{idx}\n(n={results['Sample_Size'][idx]})" for idx in results.index],
                rotation=45,
            )
            ax.legend()

        plt.tight_layout()
        plt.show()

        return results_dict

    @staticmethod
    def encoding_recommendations(
        X: pd.DataFrame,
        high_cardinality_threshold: int = 10,
        rare_category_threshold: int = 30,
        prefer_target: bool = True,
    ) -> dict:
        """
        Analyze categorical features based on cardinality and rare category to recommend encoding strategy.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        high_cardinality_threshold : int, default=10
            Number of unique values above which a column is considered high cardinality.
        rare_category_threshold : int, default=30
            Minimum samples per category, below which a category is considered rare.
        prefer_target : bool, default=True
            Whether to prefer target encoding when both encoding strategies are suitable.

        Returns
        -------
        dict
            Dictionary containing:
            - 'high_cardinality_cols': List of high cardinality columns
            - 'target_encode_cols': List of columns suitable for target encoding
            - 'onehot_encode_cols': List of columns suitable for one-hot encoding
            - 'analysis': DataFrame showing detailed analysis for each categorical column
        """
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns

        if len(categorical_cols) == 0:
            print("No categorical columns found in the dataset.")
            return {
                "high_cardinality_cols": [],
                "target_encode_cols": [],
                "onehot_encode_cols": [],
                "analysis": pd.DataFrame(),
            }

        analysis_data = []
        high_cardinality_cols = []
        target_encode_cols = []
        onehot_encode_cols = []

        for col in categorical_cols:
            print("-" * 25)
            print(f"Column `{col}` Analysis:")

            missing_rate = X[col].isna().mean()
            n_unique = X[col].nunique()
            value_counts = X[col].value_counts()
            min_category_size = value_counts.min()
            rare_categories = (value_counts < rare_category_threshold).sum()

            analysis = {
                "column": col,
                "missing_rate": missing_rate,
                "unique_values": n_unique,
                "min_category_size": min_category_size,
                "rare_categories": rare_categories,
            }

            analysis_data.append(analysis)

            # Categorize columns based on cardinality
            if n_unique > high_cardinality_threshold:
                high_cardinality_cols.append(col)
                target_encode_cols.append(col)
                if rare_categories > 0:
                    analysis["encoding_note"] = (
                        f"Has {rare_categories} rare categories. "
                        f"Recommend higher smoothing for target encoding."
                    )
                    print(
                        f"Recommend target encoding with higher smoothing parameter due to high cardinality and rare categories"
                    )
                else:
                    analysis["encoding_note"] = (
                        "Recommend target encoding due to high cardinality and no rare categories"
                    )
                    print(analysis["encoding_note"])
            elif n_unique <= 2:
                onehot_encode_cols.append(col)
                analysis["encoding_note"] = (
                    "Recommend one-hot encoding as the number of unique value is less or equal to 2"
                )
                print(analysis["encoding_note"])
            else:  # Low to moderate cardinality
                if rare_categories > 0:
                    target_encode_cols.append(col)
                    analysis["encoding_note"] = (
                        f"Has {rare_categories} rare categories, careful for sparse feature and overfitting."
                        f"Recommend higher smoothing for target encoding."
                    )
                    print(
                        """Recommend target encoding with higher smoothing due to moderate cardinality and rare category"""
                    )
                elif prefer_target:
                    target_encode_cols.append(col)
                    analysis["encoding_note"] = (
                        """Either encoding will do due to moderate cardinality and sufficient sample size, feature added to target_encode_cols based on user preference"""
                    )
                    print(analysis["encoding_note"])
                else:
                    onehot_encode_cols.append(col)
                    analysis["encoding_note"] = (
                        """Either encoding will do due to moderate cardinality and sufficient sample size, feature added to onehot_encode_cols based on user preference"""
                    )
                    print(analysis["encoding_note"])

            # Print detailed column info
            print(f"\ncolumn `{col}` details:")
            print(f"• Missing rate: {missing_rate:.1%}")
            print(f"• Unique values: {n_unique}")
            print(f"• Minimum category size: {min_category_size}")
            print(f"• Rare categories: {rare_categories}")

        analysis_df = pd.DataFrame(analysis_data)

        # Summary counts
        print("-" * 25)
        print("\nSummary:")
        print(f"• Target encoding cols: {len(target_encode_cols)}")
        print(f"• One-hot encoding cols: {len(onehot_encode_cols)}")

        return {
            "high_cardinality_cols": high_cardinality_cols,
            "target_encode_cols": target_encode_cols,
            "onehot_encode_cols": onehot_encode_cols,
            "analysis": analysis_df,
        }

    @staticmethod
    def filter_feature_selection(
        X: pd.DataFrame,
        y: pd.Series,
        task: str = "classification",
        missing_threshold: float = 0.2,
        mi_threshold: float = 0.1,
        n_top_features: int = None,
        random_state: int = 42,
    ) -> dict:
        """
        Analyzes features and recommends which ones to keep or drop based on data quality metrics.

        Parameters
        ----------
        X : pd.DataFrame
            Features to analyze.
        y : pd.Series
            Target variable.
        task : str, default="classification"
            Type of ML task - 'classification' or 'regression'.
        missing_threshold : float, default=0.2
            Maximum acceptable proportion of missing values.
        mi_threshold : float, default=0.1
            Minimum acceptable normalized mutual information score.
        n_top_features : int, optional
            If specified, selects this many top features based on mutual information scores.
            This is applied after filtering out features with too many missing values or no variance.
            If None, all features meeting the threshold criteria are kept.
        random_state : int, default=42
            Random seed for mutual information calculation.

        Returns
        -------
        dict
            Dictionary containing:
            - 'columns_to_drop': List of all columns recommended for dropping
            - 'selected_cols': List of columns recommended to keep
            - 'drop_by_missing': List of columns with too many missing values
            - 'drop_by_unique': List of columns with only one unique value
            - 'drop_by_low_mi': List of columns with low mutual information
            - 'analysis': DataFrame with detailed analysis for each feature
            - 'thresholds': Dictionary with the thresholds used for filtering

        Notes
        -----
        This method evaluates features based on three main criteria:
        1. Missing values: Features with missing values exceeding the threshold are recommended for dropping
        2. Unique values: Features with only one unique value (no variance) are recommended for dropping
        3. Mutual information: Features with low mutual information with the target are recommended for dropping

        If n_top_features is specified, the method will:
        1. First apply the basic filtering (missing values, unique values)
        2. Then select the top N features based on mutual information scores
        3. The mi_threshold will be ignored in this case

        Mutual information is calculated with either `mutual_info_classif` or `mutual_info_regression` based on
        user specified task.
        """
        # Create copy to avoid modifying original
        X_proc = X.copy()

        # Handle missing and encode categories to calculate mutual info
        discrete_features = np.array(
            [
                dtype.name in ["category", "object", "bool"]
                for col, dtype in zip(X_proc.columns, X_proc.dtypes)
            ]
        )
        cat_columns = X_proc.select_dtypes(include=["object", "category"]).columns
        for col in X_proc.columns:
            if col in cat_columns:
                most_frequent = X_proc[col].mode()[0]
                X_proc[col] = X_proc[col].fillna(most_frequent)
                X_proc[col] = pd.Categorical(X_proc[col]).codes
            else:
                X_proc[col] = X_proc[col].fillna(X_proc[col].median())

        # Calculate mutual info
        if task == "classification":
            mi_scores = mutual_info_classif(
                X_proc,
                y,
                discrete_features=discrete_features,
                random_state=random_state,
            )
        elif task == "regression":
            mi_scores = mutual_info_regression(
                X_proc,
                y,
                discrete_features=discrete_features,
                random_state=random_state,
            )
        else:
            raise ValueError("task must be 'classification' or 'regression'")

        analysis = pd.DataFrame(
            {
                "feature": X.columns,
                "missing_ratio": X.isnull().mean(),
                "n_unique": X.nunique(),
                "n_unique_ratio": X.nunique() / len(X),
                "dtype": X.dtypes,
                "is_discrete": discrete_features,
            }
        )

        # Add MI scores and feature type info to analysis
        analysis["mutual_info"] = mi_scores
        analysis["mutual_info_normalized"] = mi_scores / np.nanmax(mi_scores)
        analysis["is_discrete"] = discrete_features

        # Sort analysis by mutual_info_normalized for better readability
        analysis = (
            analysis.sort_values("mutual_info_normalized", ascending=False)
            .drop(["is_discrete"], axis=1)
            .reset_index(drop=True)
        )

        # Add reasons for dropping
        analysis["drop_missing"] = analysis["missing_ratio"] > missing_threshold
        analysis["drop_unique"] = analysis["n_unique"] <= 1

        # Get lists of columns for basic filtering
        cols_missing = analysis[analysis["drop_missing"]]["feature"].tolist()
        cols_unique = analysis[analysis["drop_unique"]]["feature"].tolist()

        # Initial filtering based on missing values and unique values
        basic_filtered_features = analysis[
            ~(analysis["drop_missing"] | analysis["drop_unique"])
        ]

        # Handle top N feature selection if specified
        if n_top_features is not None:
            if n_top_features > len(basic_filtered_features):
                print(
                    f"Warning: Requested {n_top_features} features but only {len(basic_filtered_features)} available after basic filtering"
                )
                n_top_features = len(basic_filtered_features)

            selected_features = basic_filtered_features.head(n_top_features)
            cols_low_mi = basic_filtered_features.iloc[n_top_features:][
                "feature"
            ].tolist()
            analysis["drop_mi"] = ~analysis["feature"].isin(
                selected_features["feature"]
            )
        else:
            # Use mi_threshold if n_top_features not specified
            analysis["drop_mi"] = analysis["mutual_info_normalized"] < mi_threshold
            cols_low_mi = analysis[analysis["drop_mi"]]["feature"].tolist()

        cols_to_drop = list(set(cols_missing + cols_unique + cols_low_mi))
        selected_cols = list(
            analysis[~analysis["feature"].isin(cols_to_drop)]["feature"]
        )

        # Print summary with specific columns
        print("Filter Feature Selection Summary:")
        print("==========")
        print(f"Total features analyzed: {len(X.columns)}")

        print(
            f"\n1. High missing ratio (>{missing_threshold * 100}%): {len(cols_missing)} columns"
        )
        if cols_missing:
            print("   Columns:", ", ".join(cols_missing))

        print(f"\n2. Single value: {len(cols_unique)} columns")
        if cols_unique:
            print("   Columns:", ", ".join(cols_unique))

        if n_top_features is not None:
            print(
                f"\n3. Not in top {n_top_features} features: {len(cols_low_mi)} columns"
            )
        else:
            print(
                f"\n3. Low mutual information (<{mi_threshold}): {len(cols_low_mi)} columns"
            )
        if cols_low_mi:
            print("   Columns:", ", ".join(cols_low_mi))

        print(f"\nRecommended drops: ({len(cols_to_drop)} columns in total)")

        return {
            "columns_to_drop": cols_to_drop,
            "selected_cols": selected_cols,
            "drop_by_missing": cols_missing,
            "drop_by_unique": cols_unique,
            "drop_by_low_mi": cols_low_mi,
            "analysis": analysis,
            "thresholds": {
                "missing_threshold": missing_threshold,
                "mi_threshold": mi_threshold if n_top_features is None else None,
                "n_top_features": n_top_features,
            },
        }

    @staticmethod
    def mlflow_input_prep(data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for MLflow compatibility by converting data types.

        Parameters
        ----------
        data : pd.DataFrame
            Input data possibly containing category columns.

        Returns
        -------
        pd.DataFrame
            Data with category columns converted to object dtype and integer columns to float64.

        Notes
        -----
        This function addresses two MLflow compatibility issues:
        1. MLflow doesn't support pandas category dtype, so we convert them to object
        2. MLflow doesn't handle missing values in integer columns, so we convert them to float64
        """
        if not isinstance(data, pd.DataFrame):
            return data

        data_copy = data.copy()
        # mlflow doesn't support category dtype, so we convert category to object
        for col in data_copy.select_dtypes(include=["category"]).columns:
            data_copy[col] = data_copy[col].astype("object")
        # mlflow doesn't support missing in int dtype, so we convert int to float
        for col in data_copy.select_dtypes(include=["integer"]).columns:
            data_copy[col] = data_copy[col].astype("float64")
        return data_copy
