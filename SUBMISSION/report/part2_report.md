# PART 2 - Dialogue Continuity Classification


### **Task1 Data Balancing**

You must **answer all the questions/requirements** listed below.

For this part, please choose **the same training model** (TF-IDF + SVM / XGBoost / Bert...) for comparing the two data balancing methods.

>
> For each balancing experiment I used the same model: **TF-IDF + LinearSVC**. This makes the comparison fair, as the feature extractor and the classifier are fixed, while only the balancing strategy is changed.
>
> Shared setup:
>
> | Item | Setting |
> |---|---|
> | Feature extractor | `TfidfVectorizer` |
> | Classifier | `LinearSVC` |
> | Validation | Stratified 80/20 split |
> | Random state | `42` |
> | TF-IDF settings | `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True` |

#### **Code Screenshots:** Include clear code blocks for both implementations with screenshots in your report. And, explain your code.

>
> I added one code block for the required random over-sampling method and one code block for the advanced cost-sensitive method. The same code is available in `part2/part2.ipynb` or in the script section below to take screenshots.
>
> **Code Screenshot 1 - Random over-sampling implementation**
>
> ```python
> import pandas as pd
>
> def perform_balancing(df: pd.DataFrame, method: str = "random", random_state: int = 42) -> pd.DataFrame:
>     if method == "none":
>         return df.copy()
>
>     counts = df["label"].value_counts()
>     majority_label = counts.idxmax()
>     minority_label = counts.idxmin()
>     df_majority = df[df["label"] == majority_label]
>     df_minority = df[df["label"] == minority_label]
>
>     df_minority_upsampled = df_minority.sample(
>         n=len(df_majority),
>         replace=True,
>         random_state=random_state,
>     )
>     df_balanced = pd.concat([df_majority, df_minority_upsampled], axis=0)
>     return df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
> ```
>
> This code duplicates minority-class samples with replacement until the minority class has the same number of examples as the majority class. Then it shuffles the balanced training set.
>
> **Code Screenshot 2 - Advanced balancing method: cost-sensitive LinearSVC**
>
> ```python
> from sklearn.feature_extraction.text import TfidfVectorizer
> from sklearn.metrics import f1_score
> from sklearn.svm import LinearSVC
>
> vectorizer = TfidfVectorizer(
>     ngram_range=(1, 2),
>     min_df=2,
>     sublinear_tf=True,
> )
> clf = LinearSVC(
>     C=1.0,
>     class_weight="balanced",
>     random_state=42,
>     max_iter=8000,
> )
>
> X_train = vectorizer.fit_transform(train_data["text"].astype(str))
> X_valid = vectorizer.transform(val_data["text"].astype(str))
> clf.fit(X_train, train_data["label"].astype(int))
> pred = clf.predict(X_valid)
> macro_f1 = f1_score(val_data["label"].astype(int), pred, average="macro")
> ```
>
> This code uses `class_weight="balanced"`, which weights the loss for mistakes on the minority class, rather than duplicating rows.

#### **Observations:** Please document your findings, and explain everything in details, including the listed questions.

* **The training model you choose to compare the results.**

  >
  > I chose **TF-IDF + LinearSVC**. Each dialogue is represented by a sparse word/phrase feature representation TF-IDF. LinearSVC is a linear SVM classifier . I used the same model for all balancing experiments so the comparison focuses only on the balancing method.

* **Briefly explain about random over-sampling.**

  >
  > I chose **TF-IDF + LinearSVC**. Each dialogue is represented by a sparse word/phrase feature representation TF-IDF. LinearSVC is a linear SVM classifier . I used the same model for all balancing experiments so the comparison focuses only on the balancing method.

* **Explain Macro-F1 first, give a comparison of the Macro-F1 scores resulting from those methods, and why this score.**

  >
  > Macro-F1 computes the F1 score for each class separately and averages the class-level F1 scores equally. This is suitable here because the dataset is imbalanced: the accuracy can be very high if the model mostly predicts the majority class, while Macro-F1 penalizes poor performance on either `Complete` or `Incomplete`.
  >
  > | Method | Training Model | Validation Macro-F1 | Notes |
  > |---|---|---:|---|
  > | No balancing baseline | TF-IDF + LinearSVC | **0.7919** | Best result in this comparison |
  > | Random over-sampling | TF-IDF + LinearSVC | 0.7777 | Minority rows duplicated with replacement |
  > | Cost-sensitive learning | TF-IDF + LinearSVC with `class_weight="balanced"` | 0.7771 | Minority-class mistakes receive larger penalty |

* **A determination of which method performed better, explain why.**

  >
  > The best method was the **no-balancing baseline**, with validation Macro-F1 `0.7919`. Random over-sampling likely hurt because duplicated minority examples did not add new information and may have encouraged overfitting. The cost-sensitive learning performed a little worse too, because the decision boundary might have moved too aggressively to the minority class, giving up too much precision in favor of recall.

* **An explanation of the logic behind your chosen Advanced Method.**

  >
  > The chosen advanced balancing method was **cost-sensitive learning** with `LinearSVC(class_weight="balanced")`. This method changes the loss function rather than changing the dataset . Errors on the smaller class are penalized more . The goal is that the model should pay more attention to minority examples when optimizing

* **The unique mechanism it uses to help the model learn compared to the random method.**

  >
  > Random over-sampling changes the training data by duplicating rows. Cost-sensitive learning does not alter the dataset, but alters the weighting of loss.
  >
  > | Method | Mechanism | What changes |
  > |---|---|---|
  > | Random over-sampling | Duplicates minority examples | Training dataset |
  > | Cost-sensitive learning | Increases minority-class penalty | Loss function |
  >
  > This avoids physical duplication but in this experiment still did not beat the plain baseline.

* **Best Method Identification:** Clearly state your best methods, and the specific parameters applied, explain why it outperformed and what you have tried.

  >
  > The best method for Task1 Data Balancing was the **no-balancing TF-IDF + LinearSVC baseline**.
  >
  > | Item | Value |
  > |---|---|
  > | Method | No balancing baseline |
  > | Model | TF-IDF + LinearSVC |
  > | TF-IDF parameters | `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True` |
  > | Classifier | `LinearSVC(C=1.0)` |
  > | Validation Macro-F1 | `0.7919` |
  >
  > I did not use any balancing, random oversampling and cost-sensitive `LinearSVC(class_weight="balanced")`. The baseline outperformed the balancing methods because the alternatives either repeated existing samples or shifted the decision boundary too much.

* **Any others you find interesting / like to discuss.**

  >
  > It is important to note that the inclusion of balancing did not enhance the classical model significantly. This means that the main constraint is not only unbalanced data labels but the modeling capacity of the TF-IDF + LinearSVC combination. It appears that the largest performance gain resulted from the transformer models such as RoBERTa.
---

### **Task2 Modeling & Enhancements**

You must **answer all the questions/requirements** listed below.
Note that you must at least implement one of the advanced models.


#### **Code Screenshots:** Include clear code blocks for both implementations with screenshots in your report. And, explain your code.

>
> I have incorporated both classical TF-IDF + SVM model as well as a more sophisticated one using RoBERTa. Screen-shots may be generated from either `part2/scripts/classical_oof.py`, `part2/scripts/run_oof.py` or respective  
>
> **Code Screenshot 1 - TF-IDF + SVM baseline**
>
> ```python
> import numpy as np
> from sklearn.feature_extraction.text import TfidfVectorizer
> from sklearn.metrics import f1_score
> from sklearn.model_selection import StratifiedKFold
> from sklearn.pipeline import Pipeline
> from sklearn.svm import LinearSVC
>
> CONFIGS = [
>     {
>         "name": "word_tfidf_linearsvc",
>         "vectorizer": TfidfVectorizer(
>             analyzer="word",
>             ngram_range=(1, 3),
>             min_df=2,
>             sublinear_tf=True,
>         ),
>         "model": LinearSVC(C=1.0, class_weight=None, random_state=42),
>     }
> ]
>
> y = train_df["label"].values.astype(int)
> skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
> oof_scores = np.zeros(len(train_df), dtype=np.float32)
>
> for train_idx, valid_idx in skf.split(train_df["text"], y):
>     cfg = CONFIGS[0]
>     pipe = Pipeline([("tfidf", cfg["vectorizer"]), ("svc", cfg["model"])])
>     pipe.fit(train_df.iloc[train_idx]["text"], y[train_idx])
>     valid_score = pipe.decision_function(train_df.iloc[valid_idx]["text"])
>     oof_scores[valid_idx] = valid_score.astype(np.float32)
>
> macro_f1 = f1_score(y, (oof_scores >= 0).astype(int), average="macro")
> ```
>
> This baseline converts text into TF-IDF features and trains a linear SVM decision boundary. It is fast, deterministic, and useful as the required classical comparison.
>
> **Code Screenshot 2 - Advanced RoBERTa model**
>
> ```python
> tokenizer = AutoTokenizer.from_pretrained("roberta-base")
> skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
> oof_probs = np.zeros(len(train_df), dtype=np.float32)
>
> for fold_index, (train_idx, valid_idx) in enumerate(skf.split(train_df["text"], y)):
>     fold_train = train_df.iloc[train_idx].reset_index(drop=True)
>     fold_valid = train_df.iloc[valid_idx].reset_index(drop=True)
>
>     model = AutoModelForSequenceClassification.from_pretrained(
>         "roberta-base",
>         num_labels=2,
>     ).float().to(device)
>
>     train_ds = TextDataset(fold_train["text"], fold_train["label"], tokenizer, max_length=128)
>     valid_ds = TextDataset(fold_valid["text"], fold_valid["label"], tokenizer, max_length=128)
>     train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
>     valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)
>
>     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
>     total_steps = len(train_loader) * 3
>     scheduler = get_linear_schedule_with_warmup(
>         optimizer,
>         num_warmup_steps=max(1, total_steps // 10),
>         num_training_steps=total_steps,
>     )
>
>     for epoch in range(3):
>         model.train()
>         for batch in train_loader:
>             batch = batch_to_device(batch, device, torch)
>             outputs = model(**batch)
>             outputs.loss.backward()
>             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
>             optimizer.step()
>             scheduler.step()
>             optimizer.zero_grad(set_to_none=True)
>
>     valid_probs = predict_probs(model, valid_loader, device, torch)
>     oof_probs[valid_idx] = valid_probs
>
> threshold, tuned_f1 = tune_threshold(
>     y,
>     oof_probs,
>     threshold_min=0.30,
>     threshold_max=0.70,
>     threshold_step=0.01,
> )
> ```
>
> The model fine-tunes the pre-trained RoBERTa to perform binary sequence classification. The text is tokenized using tokenizer, then RoBERTa makes context-based representation and finally, the classification head classifies as complete or incomplete. Threshold tuning is only performed on OOF validation probabilities.

#### **Observations:** Please document your findings, and explain everything in details, including:

* **A comparison of the Macro-F1 scores resulting from those methods, and explain why this score.**

  >
  > Macro-F1 was used because the dataset is imbalanced and both classes matter. It prevents a model from looking strong only because it predicts the majority class well.
  >
  > | Candidate | Validation Type | Macro-F1 | Threshold | Notes |
  > |---|---|---:|---:|---|
  > | TF-IDF + LinearSVC baseline | 80/20 split | 0.7919 | default | Classical baseline |
  > | Best classical TF-IDF + LinearSVC | 5-fold CV | 0.8015 +/- 0.0256 | default | Best classical model |
  > | DistilBERT no augmentation | OOF | 0.9531 | 0.62 | Strong transformer baseline |
  > | DistilBERT with heuristic splitting | OOF | 0.9121 | 0.30 | Augmentation hurt performance |
  > | DeBERTa-v3-base | OOF | 0.9389 | 0.46 | Below DistilBERT and RoBERTa |
  > | **RoBERTa-base seed 42** | OOF | **0.968701** | **0.510** | Final selected model |
  > | OOF LogisticRegression stacker | OOF | 0.9670 | 0.35 | Not selected because below RoBERTa |
  >
  > RoBERTa-base seed 42 achieved the highest OOF Macro-F1: `0.968701`. The Kaggle public score reported after final submission was `0.9686`, reported only as a public leaderboard score, not private test performance.

* **Briefly explain what TF-IDF and SVM is.**

  >
  > **TF-IDF** stands for Term Frequency-Inverse Document Frequency. It transforms documents into vectors such that frequent terms in one document and infrequent in the whole collection get high weight.
  >
  > **SVM** stands for Support Vector Machine. Its task is to find a hyperplane which maximizes margins between different classes. When text features are high dimensional and sparse, a linear SVM like `LinearSVC` works effectively.

* **A determination of which method performed better, explain why.**

  >
  > The winning model was **Tuned RoBERTa base seed 42**. This beat the TF-IDF + SVM model and all the other transformer models due to the fact that continuity in dialogues is based on context not just word occurrences. RoBERTa makes use of self-attention to allow each token to relate to other tokens around it thus determining continuity of an utterance.
    >
    > The stacker model was evaluated too but failed to surpass the performance of the RoBERTa model:
    >
    > ```text
    > RoBERTa OOF Macro-F1 = 0.968701
    > Stacker OOF Macro-F1 = 0.9670
    > ```
    >
    > Hence the stacker model could not be chosen.
* **An explanation of the logic behind your chosen Advanced Method.**

  >
  > The chosen advanced method was **fine-tuned RoBERTa-base for binary sequence classification**. RoBERTa is a pretrained transformer language model. It already contains general language knowledge from pretraining, and fine-tuning adapts that knowledge to the dialogue continuity labels.
  >
  > Final setup:
  >
  > | Parameter | Value |
  > |---|---|
  > | Model | `roberta-base` |
  > | Seed | `42` |
  > | Epochs | `3` |
  > | Batch size | `16` |
  > | Max length | `128` |
  > | Learning rate | `2e-5` |
  > | Weight decay | `0.01` |
  > | Threshold | `0.510` |
  >
  > Threshold tuning was performed only with OOF validation probabilities.

* **The unique mechanism it uses to help the model learn.**

  >
  > The primary component behind RoBERTa is the transformer self-attention model. Through self-attention, each token can pay attention to other tokens within the sequence, and the model learns context-based semantics as opposed to independent semantics.
  >
  > This assists the model in identifying whether the sentence is natural-ending, whether the sentence ending indicates continuity, whether there is a list involved or whether the utterance is structurally incomplete. Whereas the TF-IDF + SVM model is mostly dependent on token frequency, RoBERTa learns a much more meaningful semantic representation of the entire dialogue.

* **Best Model Identification:** Clearly state your best model, the methods used, and the specific parameters applied, explain why it outperformed and what you have tried.

  >
  > The best final model was **tuned RoBERTa-base seed 42**.
  >
  > | Item | Value |
  > |---|---|
  > | Final model | Tuned RoBERTa-base seed 42 |
  > | Final submission | `part2/submissions/submission_final.csv` |
  > | OOF Macro-F1 | `0.968701` |
  > | Threshold | `0.510` |
  > | Kaggle public score | `0.9686` |
  > | Final distribution | `{0:216, 1:284}` |
  > | Stacker OOF Macro-F1 | `0.9670` |
  > | Hamming distance vs rollback baseline | `15` |
  > | Duplicate transfer overrides | `0` |
  >
  > Methods used: Baseline of TF-IDF+LinearSVC, optimized TF-IDF+LinearSVC with 5-fold CV, random over-sampling, cost-sensitive LinearSVC, DistilBERT with no augmentation, DistilBERT with heuristic splitting, DeBERTa-v3-base, RoBERTa-base, logistic regression stacker on OOF predictions of RoBERTa, heavy probabilistic blending with RoBERTa models, and threshold tuning sweep.
  >
  > RoBERTa proved to be the best-performing model among others because of its best OOF Macro-F1 score and stability when applying threshold tuning sweep. When sweeping the range between thresholds `0.45` and `0.57`, RoBERTa achieved the same OOF score of `0.968701`, having a flat region between `0.505` and `0.521`. Our chosen threshold of `0.5

* **Any others you find interesting / like to discuss.**

  >
  > A notable observation was that the score on OOF RoBERTa validation and the Kaggle public score were virtually the same:
  >
  > ```text
  > OOF Macro-F1 = 0.968701
  > Kaggle public score = 0.9686
  > ```
  >
  > It could be considered that the validation method was sound. The public score was not used as the validation method in the tuning stage. An observation from this was that a complicated ensemble model was not necessarily better than a simpler one.



### Conclusion for Part 2

> For **Task1 Data Balancing**, the best classical result came from the simple TF-IDF + LinearSVC baseline without balancing. Random over-sampling and cost-sensitive learning were both tested, but both slightly reduced validation Macro-F1.
>
> For **Task2 Modeling & Enhancements**, the best final method was **tuned RoBERTa-base seed 42**:
>
> ```text
> OOF Macro-F1 = 0.968701
> Kaggle public score = 0.9686
> Threshold = 0.510
> ```
>
> Therefore, the final Part 2 submission is produced with **tuned RoBERTa-base seed 42**.