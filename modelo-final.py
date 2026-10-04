import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

SEED = 123
LABELS = ["c1", "c234", "c5"]
TEXT_COLUMN = "resp_text"
TARGET_COLUMN = "clarity"

def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            analyzer="word",
            max_df=0.6,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="saga",
            random_state=SEED,
        )),
    ])


def main():
    df = pd.read_excel('./train.xlsx')

    x = df[TEXT_COLUMN].astype(str)
    y = df[TARGET_COLUMN]
    distribution = y.value_counts().reindex(LABELS, fill_value=0).to_string(name=False, dtype=False)
    
    print("---------------------------------------------")
    print("-----------------  DATASET  -----------------")
    print("---------------------------------------------")
    print(f"Dataset:\n{len(df)} rows\n")
    print(f"Class distribution:\n{distribution}\n")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED, stratify=y)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_validate(
        build_pipeline(),
        x_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )["test_score"]

    print("---------------------------------------------")
    print("-----------------  TRAINING -----------------")
    print("---------------------------------------------")
    print(f"Cross-validation macro-F1: {scores.mean():.4f} +/- {scores.std():.4f}")
    print(f"Fold scores: {scores}\n")

    model = build_pipeline()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    print("---------------------------------------------")
    print("-----------------  TESTING  -----------------")
    print("---------------------------------------------")
    print("Test macro-F1:")
    print(f1_score(y_test, predicted, average='macro'))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predicted, labels=LABELS))
    print("\nClassification report:")
    print(classification_report(y_test, predicted, labels=LABELS, target_names=LABELS, zero_division=0))


if __name__ == "__main__":
    main()