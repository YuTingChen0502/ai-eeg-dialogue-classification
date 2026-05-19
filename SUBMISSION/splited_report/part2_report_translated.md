# PART 2 - Dialogue Continuity Classification


### **Task1 Data Balancing**

You must **answer all the questions/requirements** listed below.

For this part, please choose **the same training model** (TF-IDF + SVM / XGBoost / Bert...) for comparing the two data balancing methods.

>
> 在每個平衡實驗中，我都使用同一個模型：**TF-IDF + LinearSVC**。這使比較公平，因為特徵抽取器與分類器都是固定的，只有平衡策略不同。
>
> 共用設定：
>
> | 項目 | 設定 |
> |---|---|
> | Feature extractor（特徵抽取器） | `TfidfVectorizer` |
> | Classifier（分類器） | `LinearSVC` |
> | Validation（驗證方式） | Stratified 80/20 split（分層 80/20 切分） |
> | Random state（隨機種子） | `42` |
> | TF-IDF 設定 | `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True` |

#### **Code Screenshots:** Include clear code blocks for both implementations with screenshots in your report. And, explain your code.

>
> 我為要求的 random over-sampling 方法新增了一個程式碼區塊，並為 advanced cost-sensitive 方法新增了另一個程式碼區塊。相同的程式碼可在 `part2/part2.ipynb` 或下方的 script 區段中找到以便截圖。
>
> **Code Screenshot 1 - Random over-sampling 實作**
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
> 此程式碼以「有放回抽樣」的方式複製少數類別樣本，直到少數類別的樣本數與多數類別相同，接著將平衡後的訓練集打亂。
>
> **Code Screenshot 2 - 進階平衡方法：cost-sensitive LinearSVC**
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
> 此程式碼使用 `class_weight="balanced"`，會對少數類別的錯誤分類加重損失權重，而非直接複製資料列。

#### **Observations:** Please document your findings, and explain everything in details, including the listed questions.

* **The training model you choose to compare the results.**

  >
  > 我選擇了 **TF-IDF + LinearSVC**。每段對話以稀疏的詞／詞組 TF-IDF 特徵表示。LinearSVC 是一個線性 SVM 分類器。我在所有平衡實驗中都使用相同的模型，使比較專注於平衡方法本身。

* **Briefly explain about random over-sampling.**

  >
  > 我選擇了 **TF-IDF + LinearSVC**。每段對話以稀疏的詞／詞組 TF-IDF 特徵表示。LinearSVC 是一個線性 SVM 分類器。我在所有平衡實驗中都使用相同的模型，使比較專注於平衡方法本身。

* **Explain Macro-F1 first, give a comparison of the Macro-F1 scores resulting from those methods, and why this score.**

  >
  > Macro-F1 會分別計算每個類別的 F1 分數，然後對各類別的 F1 等權平均。這在本場景中非常適合，因為資料集是不平衡的：若模型大多預測多數類別，accuracy 可能會非常高，而 Macro-F1 會懲罰在 `Complete` 或 `Incomplete` 任一類別上表現不佳的模型。
  >
  > | 方法 | 訓練模型 | Validation Macro-F1 | 備註 |
  > |---|---|---:|---|
  > | No balancing baseline（不做平衡的基線） | TF-IDF + LinearSVC | **0.7919** | 本次比較中的最佳結果 |
  > | Random over-sampling | TF-IDF + LinearSVC | 0.7777 | 對少數類別資料有放回複製 |
  > | Cost-sensitive learning | TF-IDF + LinearSVC with `class_weight="balanced"` | 0.7771 | 對少數類別錯誤施加較重懲罰 |

* **A determination of which method performed better, explain why.**

  >
  > 最佳的方法是 **no-balancing baseline**，validation Macro-F1 為 `0.7919`。Random over-sampling 之所以反而受損，可能是因為複製的少數類別樣本並未提供新資訊，反而可能助長 overfitting。Cost-sensitive learning 也略差，因為決策邊界可能過度偏向少數類別，犧牲了太多 precision 以換取 recall。

* **An explanation of the logic behind your chosen Advanced Method.**

  >
  > 我選擇的 advanced 平衡方法是使用 `LinearSVC(class_weight="balanced")` 的 **cost-sensitive learning**。此方法修改的是 loss function，而非資料集本身。對於少數類別的錯誤，會施加較大的懲罰。其目的在於讓模型在最佳化時，對少數類別樣本投入更多關注。

* **The unique mechanism it uses to help the model learn compared to the random method.**

  >
  > Random over-sampling 透過複製資料列來改變訓練資料；cost-sensitive learning 不改動資料集，而是調整 loss 的權重。
  >
  > | 方法 | 機制 | 變動的部分 |
  > |---|---|---|
  > | Random over-sampling | 複製少數類別樣本 | 訓練資料集 |
  > | Cost-sensitive learning | 對少數類別加重懲罰 | Loss function |
  >
  > 此方法避免了實體上的資料複製，但在本實驗中仍無法超越單純的 baseline。

* **Best Method Identification:** Clearly state your best methods, and the specific parameters applied, explain why it outperformed and what you have tried.

  >
  > Task1 Data Balancing 的最佳方法為 **no-balancing TF-IDF + LinearSVC baseline**。
  >
  > | 項目 | 值 |
  > |---|---|
  > | 方法 | No balancing baseline |
  > | 模型 | TF-IDF + LinearSVC |
  > | TF-IDF 參數 | `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True` |
  > | 分類器 | `LinearSVC(C=1.0)` |
  > | Validation Macro-F1 | `0.7919` |
  >
  > 我嘗試過：不做任何平衡、random oversampling、以及 cost-sensitive `LinearSVC(class_weight="balanced")`。Baseline 勝出，是因為其他方法要嘛重複既有樣本，要嘛使決策邊界偏移過多。

* **Any others you find interesting / like to discuss.**

  >
  > 值得注意的是，加入平衡策略並未顯著提升傳統模型的表現。這意味著主要的瓶頸不僅來自資料標籤的不平衡，更來自 TF-IDF + LinearSVC 這個組合的建模能力上限。看起來，最大的效能提升來自 transformer 模型（例如 RoBERTa）。
---

### **Task2 Modeling & Enhancements**

You must **answer all the questions/requirements** listed below.
Note that you must at least implement one of the advanced models.


#### **Code Screenshots:** Include clear code blocks for both implementations with screenshots in your report. And, explain your code.

>
> 我同時實作了傳統的 TF-IDF + SVM 模型，以及更進階的 RoBERTa 模型。截圖可由 `part2/scripts/classical_oof.py`、`part2/scripts/run_oof.py` 或對應檔案產生。
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
> 此 baseline 將文字轉成 TF-IDF 特徵，再訓練一條線性 SVM 決策邊界。它快速、決定性高，且作為要求中的傳統模型對照非常合適。
>
> **Code Screenshot 2 - 進階 RoBERTa 模型**
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
> 此模型對預訓練的 RoBERTa 進行 fine-tuning 來執行二元序列分類。文字先經 tokenizer 進行 tokenization，接著由 RoBERTa 產生具上下文資訊的表示，最後由 classification head 預測 complete 或 incomplete。Threshold tuning 僅在 OOF validation 機率上進行。

#### **Observations:** Please document your findings, and explain everything in details, including:

* **A comparison of the Macro-F1 scores resulting from those methods, and explain why this score.**

  >
  > 之所以採用 Macro-F1，是因為資料集不平衡而且兩個類別都同樣重要。它避免模型僅因為多數類別預測得好而被誤判為強。
  >
  > | 候選模型 | Validation 類型 | Macro-F1 | Threshold | 備註 |
  > |---|---|---:|---:|---|
  > | TF-IDF + LinearSVC baseline | 80/20 split | 0.7919 | default | 傳統基線 |
  > | Best classical TF-IDF + LinearSVC | 5-fold CV | 0.8015 +/- 0.0256 | default | 最佳傳統模型 |
  > | DistilBERT no augmentation | OOF | 0.9531 | 0.62 | 表現強的 transformer baseline |
  > | DistilBERT with heuristic splitting | OOF | 0.9121 | 0.30 | augmentation 反而降低表現 |
  > | DeBERTa-v3-base | OOF | 0.9389 | 0.46 | 低於 DistilBERT 與 RoBERTa |
  > | **RoBERTa-base seed 42** | OOF | **0.968701** | **0.510** | 最終選用的模型 |
  > | OOF LogisticRegression stacker | OOF | 0.9670 | 0.35 | 因低於 RoBERTa 而未被選用 |
  >
  > RoBERTa-base seed 42 達到最高的 OOF Macro-F1：`0.968701`。最終 submission 後回報的 Kaggle public score 為 `0.9686`，此分數僅為 public leaderboard 分數，並非 private test 表現。

* **Briefly explain what TF-IDF and SVM is.**

  >
  > **TF-IDF** 為 Term Frequency-Inverse Document Frequency。它將文件轉成向量，使得在單一文件中常見、但在整體語料中罕見的詞，獲得較高的權重。
  >
  > **SVM** 為 Support Vector Machine。它的任務是找出一個能最大化不同類別之間 margin 的 hyperplane。當文字特徵為高維且稀疏時，像 `LinearSVC` 這種線性 SVM 會非常有效。

* **A determination of which method performed better, explain why.**

  >
  > 勝出的模型是 **Tuned RoBERTa base seed 42**。它擊敗了 TF-IDF + SVM 模型以及其他所有 transformer 模型，原因在於：對話的連續性取決於 context，而不只是詞彙出現頻率。RoBERTa 透過 self-attention，使每個 token 能與其周圍的 token 相關聯，從而判斷某段話語的連續性。
    >
    > 我們也評估了 stacker 模型，但仍無法超越 RoBERTa：
    >
    > ```text
    > RoBERTa OOF Macro-F1 = 0.968701
    > Stacker OOF Macro-F1 = 0.9670
    > ```
    >
    > 因此最終並未選擇 stacker 模型。
* **An explanation of the logic behind your chosen Advanced Method.**

  >
  > 所選的 advanced method 是 **fine-tuned RoBERTa-base for binary sequence classification**。RoBERTa 是一個預訓練的 transformer 語言模型。它在 pretraining 階段已具備一般語言知識，而 fine-tuning 則讓這些知識適應對話連續性分類的標籤。
  >
  > 最終設定：
  >
  > | 參數 | 值 |
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
  > Threshold tuning 僅在 OOF validation 機率上執行。

* **The unique mechanism it uses to help the model learn.**

  >
  > RoBERTa 的核心是 transformer 的 self-attention 機制。透過 self-attention，每個 token 可以關注序列中的其他 token，使模型能學到 context-based 的語意，而不是孤立的語意。
  >
  > 這幫助模型辨識：該句是否自然結束、句尾是否暗示連續、是否涉及列舉、或該段話語在結構上是否不完整。相較之下，TF-IDF + SVM 主要依賴 token 頻率，而 RoBERTa 則學習到對整段對話更具語意意義的表示。

* **Best Model Identification:** Clearly state your best model, the methods used, and the specific parameters applied, explain why it outperformed and what you have tried.

  >
  > 最終最佳模型為 **tuned RoBERTa-base seed 42**。
  >
  > | 項目 | 值 |
  > |---|---|
  > | 最終模型 | Tuned RoBERTa-base seed 42 |
  > | 最終 submission | `part2/submissions/submission_final.csv` |
  > | OOF Macro-F1 | `0.968701` |
  > | Threshold | `0.510` |
  > | Kaggle public score | `0.9686` |
  > | 最終預測分布 | `{0:216, 1:284}` |
  > | Stacker OOF Macro-F1 | `0.9670` |
  > | 與 rollback baseline 的 Hamming distance | `15` |
  > | Duplicate transfer overrides | `0` |
  >
  > 嘗試過的方法包含：TF-IDF+LinearSVC 的 baseline、含 5-fold CV 的最佳化 TF-IDF+LinearSVC、random over-sampling、cost-sensitive LinearSVC、無 augmentation 的 DistilBERT、含 heuristic splitting 的 DistilBERT、DeBERTa-v3-base、RoBERTa-base、在 RoBERTa OOF 預測上的 logistic regression stacker、含 RoBERTa 模型的 heavy probabilistic blending，以及 threshold tuning sweep。
  >
  > 在眾多方法之中，RoBERTa 證明是表現最佳的，因為它擁有最佳的 OOF Macro-F1 分數，且在 threshold tuning sweep 中保持穩定。當在 `0.45` 到 `0.57` 之間掃描 threshold 時，RoBERTa 達到相同的 OOF 分數 `0.968701`，在 `0.505` 到 `0.521` 之間存在一個平坦區域。我們選擇的 threshold 為 `0.5`。

* **Any others you find interesting / like to discuss.**

  >
  > 一個值得注意的觀察是：OOF RoBERTa validation 分數與 Kaggle public score 幾乎完全一致：
  >
  > ```text
  > OOF Macro-F1 = 0.968701
  > Kaggle public score = 0.9686
  > ```
  >
  > 由此可以認為，所採用的 validation 方法是穩健的。Public score 並未被用作 tuning 階段的驗證手段。從中也可看出：複雜的 ensemble 模型不一定比較單純的模型更好。



### Conclusion for Part 2

> 對於 **Task1 Data Balancing**，最佳的傳統模型結果來自於不做平衡的單純 TF-IDF + LinearSVC baseline。Random over-sampling 與 cost-sensitive learning 都有測試，但兩者都略微降低了 validation Macro-F1。
>
> 對於 **Task2 Modeling & Enhancements**，最終最佳方法為 **tuned RoBERTa-base seed 42**：
>
> ```text
> OOF Macro-F1 = 0.968701
> Kaggle public score = 0.9686
> Threshold = 0.510
> ```
>
> 因此，Part 2 的最終 submission 是以 **tuned RoBERTa-base seed 42** 產生。
