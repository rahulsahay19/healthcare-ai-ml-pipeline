from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# This function returns the raw input feature columns for the selected model.
def get_input_feature_columns(config: dict) -> list[str]:
    return config["input_features"]


# This function builds the preprocessing pipeline dynamically using numeric and categorical
# feature lists from MODEL_CONFIG.
def get_preprocessor(config: dict) -> ColumnTransformer:
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


# This function builds the complete sklearn pipeline:
# preprocessing + RandomForest model.
def build_model_pipeline(config: dict) -> Pipeline:
    preprocessor = get_preprocessor(config)

    model = RandomForestClassifier(
        n_estimators=config["params"]["n_estimators"],
        max_depth=config["params"]["max_depth"],
        min_samples_split=config["params"]["min_samples_split"],
        min_samples_leaf=config["params"]["min_samples_leaf"],
        class_weight=config["params"]["class_weight"],
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline