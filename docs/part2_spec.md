## PART 2: Dialogue Continuity Classification

## Mission Background

In developing voice assistants (e.g., Siri or Alexa), the system must determine if a user has
finished speaking ( **Complete** ) or if the utterance is unfinished/interrupted ( **Incomplete** ).

Robust endpoint detection **facilitates seamless, human-like interaction** by precisely
identifying when a user has finished speaking, thereby eliminating awkward silences and
preventing premature processing of incomplete thoughts.

This mechanism ensures conversational flow and barge-in capabilities while optimizing
computational efficiency by triggering resource-intensive LLM inference only when a
complete, actionable request is finalized.

This assignment uses a dataset containing transcriptions of real-world dialogues.

```
● Label 1 (Complete): Semantic intent is finished; the system can respond.
● Label 0 (Incomplete): Semantic intent is unfinished; the system should continue
listening.
```
# We provide two .CSV file:

```
● train.csv -> model training
● test.csv -> kaggle competition
```
train.csv for training your model. test.csv for generating your solution.csv for Kaggle
competition. Please use the train.csv and test.csv file to complete part 2.

## 1. Data Balancing (15% Total)

The dataset is naturally imbalanced. You must demonstrate how different sampling
techniques affect the model’s ability to recognize the minority class.

**Code Implementation (8%)**

Please complete your code using part2_code_template.ipynb.


Since the dataset is imbalanced, models may bias toward the majority class. You must
implement and compare two methods. For this part, please choose **the same training
model** (TF-IDF + SVM/ XGBoost/ Bert....) for comparing the two data balancing methods :

```
● Basic Method (Required): Random Over-sampling. Duplicate minority class
samples until the ratio is 1:.
● Advanced Method (Choose 1+):
○ EDA (Easy Data Augmentation): Random synonym replacement, insertion,
swapping, or deletion.
○ Back-translation: Translate text to another language (e.g., German) and
back to English.
○ SMOTE: Synthesize minority samples in the vector space.
○ Cost-Sensitive Learning: Adjust class_weight='balanced' in the model loss
function.
○ Heuristic Splitting: Split complete sentences into incomplete sentences.
○ Any other method you would like to use.
```
**Written Report (7%)**

You must **answer all the questions/requirements** listed below**.**

For this part, please choose **the same training model** (TF-IDF + SVM/ XGBoost/ Bert....)
for comparing the two data balancing methods.

```
● Code Screenshots: Include clear code blocks for both implementations with
screenshots in your report. And, explain your code.
● Observations: Please document your findings, and explain everything in details,
including
○ The training model you choose to compare the results.
○ Briefly explain about random over-sampling.
○ Explain Macro-F1 first, give a comparison of the Macro-F1 scores resulting
from those methods, and why this score.
○ A determination of which method performed better, explain why.
○ An explanation of the logic behind your chosen "Advanced Method."
○ The unique mechanism it uses to help the model learn compared to the
random method.
○ Best Method Identification: Clearly state your best methods, and the
specific parameters applied, explain why it outperformed and what you have
tried.
○ Any others you find interesting/ like to discuss.
```
## 2. Modeling & Enhancements (25% Total)

Please complete your code using part2_code_template.ipynb.


You are required to move from a traditional machine learning baseline to a more
sophisticated architecture.

**Code Implementation (10%)**

1. **Baseline Model:** Establish a stable baseline using **TF-IDF + SVM (Support Vector**
    **Machine)**.
       ○ **Metric:** Macro-F1 Score (Do not use Accuracy due to class imbalance).
2. **Advanced Method:** Implement more advanced methods to improve your model
    accuracy. You may explore the following aspects:
       ○ **Rigor:** Use Stratified K-Fold for validation.
       ○ **Standardization:** Experiment with Lemmatization or N-grams.
       ○ **Advanced Models:** Try XGBoost or fine-tune BERT/RoBERTa or other
          models you like for a significant score boost.
       ○ Note that, you don't have to try all of them,but you must **at least implement**
          **one of the advanced models**.

**Written Report (15%)**

You must **answer all the questions/requirements** listed below.
Note that you must at least implement one of the advanced models.

```
● Code Screenshots: Include clear code blocks for both implementations with
screenshots in your report. And, explain your code.
● Observations: Please document your findings,and explain everything in details,
including:
○ A comparison of the Macro-F1 scores resulting from those methods, and
explain why this score.
○ Briefly explain what TF-IDF and SVM is.
○ A determination of which method performed better, explain why..
○ An explanation of the logic behind your chosen "Advanced Method."
○ The unique mechanism it uses to help the model learn.
○ Best Model Identification: Clearly state your best model, the methods used,
and the specific parameters applied., explain why it outperformed and what
you have tried.
○ Any others you find interesting/ like to discuss.
```
## 3. Kaggle Benchmarking (10% Total)

Your final model must be tested against the private leaderboard to validate its real-world
performance.

**Benchmarking (6%)**

```
● Requirement: Your submission must achieve a Macro-F1 score that exceeds the
established TA Baseline.
```

```
None
```
**Top Tier Bonus (4%)**

```
● Criteria: Additional marks (up to 4%) will be awarded based on your relative standing
on the leaderboard.
```
## 4. Peer Evaluation

For the peer evaluation table, each group may rate member contributions on a 1-10 scale.
You may decide the score based on your own group collaboration, but the table should
briefly indicate each member's contribution.

## 5. Submission Format Requirements

To ensure your assignment is graded, please follow these formatting rules:

**File Format:**

```
● Submit your code and report to E3 separately.
● Please note that each group should upload only one set of files in total.
● Report: Submit in .pdf format.
○ Filename: CLASSNUMBER_GROUPNUMBER_HW3_report.pdf
○ For examples: 515512_Group01_HW3_report.pdf
○ Please refer to E3 to check if you're from class 515512 or 515513.
○ Requirement: Include all members' student IDs and names at the very
beginning of the report.
○ You should only submit ONE report for the entire homework.
● Code: For Part 2 code, submit in .ipynb format.
○ Filename: part2.ipynb
○ For examples: part2.ipynb
● Code_Entire: For entire code, submit in .zip format.
○ Filename: CLASSNUMBER_GROUPNUMBER_HW3_code.zip
○ For examples: 515512_Group01_HW3_code.zip
```
Please follow the submission layout:

```
{515512|515513}_{groupNum}_HW3_code.zip
└─ part1/
├─ task1/
│ └─ task1.ipynb
└─ task2/
├─ inference.py
├─ train_command.txt
├─ requirements.txt
```

```
└─ (additional Task 2 files)
└─ part2/
└─ part2.ipynb
```
```
{515512|515513}_{groupNum}_HW3_report.pdf
```
```
● Incorrect file formats will result in a score of 0 for this homework.
● Doesn’t include all members' student ids and names at the very beginning for both
code and report will result in a score of 0 for this homework.
```
**Kaggle Submission Format:**

```
● Submit your result to Kaggle.
● Please note that each group should only have one group name in total.
● Results: Submit in .csv format.
● Group Name: CLASSNUMBER_GROUPNUMBER
○ For example: 515512_01.
○ Please refer to E3 to check if you're from class 515512 or 515513.
● Incorrect group name will result in a score of 0 for this homework.
```
**Submission Deadline:**

```
● Upload to E3 by: 2026/05/19 (Tue) 23:
● Upload to Kaggle by: 2026/05/19 (Tue) 23:
● Late submissions for any reason are not allowed. Late submissions are judged
by the E3 system. Please upload your homework early.
● Late submissions will result in a score of 0 for this homework.
```
You can find all file you need for part 2 here:

```
❖ train.csv: train.csv
❖ test.csv: test.csv
❖ part2 code template: part2_code_template.ipynb
❖ report template: HW3_report_template
❖ part 2 kaggle: https://www.kaggle.com/competitions/dialogue-continuity-classification
❖ Invitation link : https://www.kaggle.com/t/be85b84fe61e5d3e74c10da
❖ part 2 introduction: part2_introduction
```

