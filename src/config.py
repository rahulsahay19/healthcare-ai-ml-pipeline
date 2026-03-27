MODEL_CONFIG = {
    "risk": {
        "run_name": "RandomForest-Risk",
        "registered_model_name": "HealthcareRiskRFModel",

        # Core columns
        "target_column": "risk_score",
        "sort_column": "visit_date",
        "positive_label_for_recall": "High",

        # Input features
        "input_features": [
            "age",
            "gender",
            "city",
            "insurance_provider",
            "chronic_flag",
            "department",
            "visit_type",
            "doctor_id",
            "length_of_stay_hours",
            "days_since_registration",
            "visit_frequency",
            "avg_los_per_patient",
            "visit_month",
            "visit_dayofweek"
        ],

        # Feature types
        "numeric_features": [
            "age",
            "chronic_flag",
            "length_of_stay_hours",
            "days_since_registration",
            "visit_frequency",
            "doctor_id",
            "avg_los_per_patient",
            "visit_month",
            "visit_dayofweek"
        ],

        "categorical_features": [
            "gender",
            "city",
            "insurance_provider",
            "department",
            "visit_type"
        ],

        # Model thresholds
        "promotion_accuracy_threshold": 0.55,
        "promotion_recall_threshold": 0.70,

        # Local storage
        "local_model_file": "risk_model_complete_pipeline.joblib",

        # Model parameters
        "params": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample"
        }
    },

    "claim": {
        "run_name": "RandomForest-Claim",
        "registered_model_name": "HealthcareClaimRFModel",

        # Core columns
        "target_column": "claim_status",
        "sort_column": "billing_date",
        "positive_label_for_recall": "Rejected",

        # Input features
        "input_features": [
            "age",
            "gender",
            "city",
            "insurance_provider",
            "chronic_flag",
            "department",
            "visit_type",
            "doctor_id",
            "length_of_stay_hours",
            "risk_score",
            "billed_amount",
            "days_since_registration",
            "visit_frequency",
            "avg_los_per_patient",
            "provider_rejection_rate",
            "visit_month",
            "visit_dayofweek",
            "high_cost_visit_flag"
        ],

        # Feature types
        "numeric_features": [
            "age",
            "chronic_flag",
            "length_of_stay_hours",
            "doctor_id",
            "billed_amount",
            "days_since_registration",
            "visit_frequency",
            "avg_los_per_patient",
            "provider_rejection_rate",
            "visit_month",
            "visit_dayofweek",
            "high_cost_visit_flag"
        ],

        "categorical_features": [
            "gender",
            "city",
            "insurance_provider",
            "department",
            "visit_type",
            "risk_score"
        ],

        # Model thresholds
        "promotion_accuracy_threshold": 0.55,
        "promotion_recall_threshold": 0.70,

        # Local storage
        "local_model_file": "claim_model_complete_pipeline.joblib",

        # Model parameters
        "params": {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample"
        }
    }
}