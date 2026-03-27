from sklearn.metrics import accuracy_score, f1_score, recall_score

# recall_score needs the positive label specified for binary classification, 
# and we set average=None to get recall for each class. We then take the first element [0] since we
# are interested in the recall of the positive class.
def evaluate_model(y_test, predictions, positive_label: str) -> dict: 
    accuracy = accuracy_score(y_test, predictions)
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    target_recall = recall_score(
        y_test,
        predictions,
        labels=[positive_label],
        average=None,
        zero_division=0
    )[0]

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "target_recall": target_recall
    }

# This function checks if the model meets the specified thresholds for accuracy and recall to be eligible
# for production promotion.
def is_eligible_for_production(
    accuracy: float,
    target_recall: float,
    accuracy_threshold: float,
    recall_threshold: float
) -> bool:
    return (
        accuracy >= accuracy_threshold and
        target_recall >= recall_threshold
    )