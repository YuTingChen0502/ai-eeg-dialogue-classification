# Final Phase Summary

## Baseline
- File: `submission_bert_common_top3_seed_average.csv`
- Common-split Macro-F1: `0.9453`
- Threshold: `0.48`
- Distribution: `{0: 217, 1: 283}`

## Preflight
- Python executable: `C:\Coding\Intro_to_AI\project1-part2\.venv-part2-cuda\Scripts\python.exe`
- train.csv shape: `(1302, 3)`
- test.csv shape: `(500, 2)`
- train label distribution: `{0: 465, 1: 837}`

## Heuristic Splitting Ablation
# Heuristic Splitting Ablation

| tag | augmentation | tuned OOF Macro-F1 | threshold | distribution |
| --- | --- | ---: | ---: | --- |
| distilbert_no_aug | no | 0.9531 | 0.62 | {'0': 463, '1': 839} |
| distilbert | yes | 0.9121 | 0.30 | {'0': 529, '1': 773} |

Delta (aug - no_aug): `-0.0410`.
Decision: **heuristic splitting clearly hurt DistilBERT OOF; disable for later final transformer runs**.

Thresholds were tuned only on OOF probabilities. No test labels or public leaderboard signals were used.


## Transformer OOF Results
- `distilbert_no_aug`: default `0.9496`, tuned `0.9531`, threshold `0.62`, folds `[0.9539, 0.9535, 0.9531, 0.9667, 0.9207]`, std `0.0153`
- `distilbert`: default `0.8946`, tuned `0.9121`, threshold `0.30`, folds `[0.9349, 0.9101, 0.8533, 0.9035, 0.8722]`, std `0.0288`
- `deberta_v3`: default `0.9370`, tuned `0.9389`, threshold `0.46`, folds `[0.9506, 0.9338, 0.9379, 0.9583, 0.9040]`, std `0.0187`
- `roberta`: default `0.9678`, tuned `0.9687`, threshold `0.51`, folds `[0.9705, 0.9702, 0.9660, 0.9790, 0.9531]`, std `0.0085`

## Classical TF-IDF/SVM
- Meta: `{'average_score_default_macro_f1': 0.8117377832387747, 'average_score_threshold': -0.05719305276870723, 'average_score_tuned_macro_f1': 0.8194393580353296, 'feature_names': ['word_tfidf_linearsvc', 'char_wb_tfidf_linearsvc'], 'models': [{'fold_macro_f1_tuned': [0.7844827586206897, 0.7893948126801152, 0.8138098951288213, 0.8538645597469128, 0.7902521008403361], 'name': 'word_tfidf_linearsvc', 'oof_macro_f1_default_threshold_zero': 0.8021705352612887, 'oof_macro_f1_tuned': 0.8023924594240044, 'oof_threshold': 0.00493738055229187, 'prediction_distribution_at_threshold': {'0': 413, '1': 889}}, {'fold_macro_f1_tuned': [0.8162442396313364, 0.7892546583850932, 0.8268886043533932, 0.8593073593073592, 0.796078431372549], 'name': 'char_wb_tfidf_linearsvc', 'oof_macro_f1_default_threshold_zero': 0.8003702185315893, 'oof_macro_f1_tuned': 0.8036692636340789, 'oof_threshold': -0.008730581998825038, 'prediction_distribution_at_threshold': {'0': 409, '1': 893}}], 'runtime_seconds': 9.07991075515747, 'shape_test': [500, 2], 'shape_train': [1302, 2], 'validation': '5-fold StratifiedKFold on train only; test scores averaged across folds.'}`

## Endpoint Features
- Correlations file: `C:\Coding\Intro_to_AI\project1-part2\part2\outputs\features\endpoint_correlations.csv`

## Retrieval Features
- Meta: `{'feature_names': ['nn_max_sim_word', 'nn_max_sim_char', 'top3_label1_mean_word', 'top5_label1_mean_word', 'top3_label1_mean_char', 'top5_label1_mean_char', 'top5_label_agreement_word', 'top5_label_agreement_char', 'exact_duplicate_flag', 'exact_duplicate_label_if_any', 'exact_duplicate_count', 'near_duplicate_099_flag', 'near_duplicate_099_label_if_any', 'near_duplicate_099_similarity'], 'runtime_seconds': 4.307729005813599, 'shape_test': [500, 14], 'shape_train': [1302, 14], 'test_exact_duplicate_rows': 13, 'test_near_duplicate_099_rows': 14, 'test_retrieval': 'full train retrieval; labels used only as stacker features', 'train_exact_duplicate_rows': 34, 'train_retrieval': "out-of-fold; each validation row retrieves only from that fold's training rows"}`

## Stacker
- Metadata: `{'best_single_model': {'oof_macro_f1_tuned': 0.9687014960573137, 'tag': 'roberta', 'threshold': 0.5100000000000002}, 'candidates': [{'C': 0.3, 'default_f1': 0.9636260629855267, 'fold_macro_f1_default': [0.9664265500385902, 0.9659001829108962, 0.957516339869281, 0.9790237999193223, 0.9490196078431372], 'threshold': 0.5400000000000003, 'tuned_f1': 0.9644901712243371}, {'C': 1.0, 'default_f1': 0.9636632689817655, 'fold_macro_f1_default': [0.9706953244045232, 0.9659001829108962, 0.957516339869281, 0.9748889318137918, 0.9490196078431372], 'threshold': 0.35000000000000003, 'tuned_f1': 0.9669756990624969}, {'C': 3.0, 'default_f1': 0.9636632689817655, 'fold_macro_f1_default': [0.9706953244045232, 0.9659001829108962, 0.9616638542571143, 0.9748889318137918, 0.9449143564921201], 'threshold': 0.3, 'tuned_f1': 0.9669756990624969}], 'chosen_oof_macro_f1': 0.9687014960573137, 'chosen_source': 'fallback:roberta', 'chosen_threshold': 0.5100000000000002, 'fallback_used': True, 'feature_names': ['transformer:distilbert_no_aug', 'transformer:deberta_v3', 'transformer:roberta', 'classical:word_tfidf_linearsvc', 'classical:char_wb_tfidf_linearsvc', 'endpoint:ends_with_prep', 'endpoint:ends_with_det', 'endpoint:ends_with_conj', 'endpoint:ends_with_aux', 'endpoint:ends_with_wh', 'endpoint:ends_with_comma', 'endpoint:ends_with_dash', 'endpoint:ends_with_ellipsis', 'endpoint:ends_with_question', 'endpoint:ends_with_period', 'endpoint:ends_with_exclaim', 'endpoint:no_terminal_punct', 'endpoint:starts_with_request', 'endpoint:has_filler', 'endpoint:word_count', 'endpoint:char_count', 'endpoint:log1p_word_count', 'endpoint:short_utterance', 'endpoint:medium_utterance', 'endpoint:long_utterance', 'endpoint:last_token_is_incomplete_group', 'endpoint:last_token_is_function_word', 'endpoint:last_token_len', 'endpoint:comma_count', 'endpoint:question_count', 'endpoint:terminal_punct_present', 'retrieval:nn_max_sim_word', 'retrieval:nn_max_sim_char', 'retrieval:top3_label1_mean_word', 'retrieval:top5_label1_mean_word', 'retrieval:top3_label1_mean_char', 'retrieval:top5_label1_mean_char', 'retrieval:top5_label_agreement_word', 'retrieval:top5_label_agreement_char', 'retrieval:exact_duplicate_flag', 'retrieval:exact_duplicate_label_if_any', 'retrieval:exact_duplicate_count', 'retrieval:near_duplicate_099_flag', 'retrieval:near_duplicate_099_label_if_any', 'retrieval:near_duplicate_099_similarity'], 'hamming_vs_rollback_baseline': 15, 'notes': ['DistilBERT ablation selected no-augmentation version.', 'Loaded classical features with shape (1302, 2).', 'Loaded endpoint features with shape (1302, 26).', 'Loaded retrieval features with shape (1302, 14).'], 'prediction_distribution': {'0': 216, '1': 284}, 'runtime_seconds': 1.2959308624267578, 'scale_columns': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44], 'selected_C': 1.0, 'stacker_oof_macro_f1_default': 0.9636632689817655, 'stacker_oof_macro_f1_tuned': 0.9669756990624969, 'stacker_oof_threshold': 0.35000000000000003, 'submission_validation': {'distribution': {'0': 216, '1': 284}, 'errors': [], 'path': 'C:\\Coding\\Intro_to_AI\\project1-part2\\part2\\submissions\\submission_stacker_oof.csv', 'shape': [500, 2], 'valid': True}, 'x_test_shape': [500, 45], 'x_train_shape': [1302, 45]}`

## Duplicate Transfer
- Metadata: `{'aborted_due_to_override_guard': False, 'applied_override_count': 0, 'base_input': 'C:\\Coding\\Intro_to_AI\\project1-part2\\part2\\submissions\\submission_stacker_oof.csv', 'changed_ids': [], 'conflicting_train_group_rows': 0, 'diagnostic_exact_match_rows': 13, 'distribution_after': {'0': 216, '1': 284}, 'distribution_before': {'0': 216, '1': 284}, 'final_output': 'C:\\Coding\\Intro_to_AI\\project1-part2\\part2\\submissions\\submission_final.csv', 'safe_override_rule': ['test normalized text exactly matches a train group', 'train group count >= 2', 'train group labels are unanimous', 'override differs from current prediction']}`

## Final Selection
- Final selected file: `submissions\submission_final.csv`
- CSV validation: `{'path': 'C:\\Coding\\Intro_to_AI\\project1-part2\\part2\\submissions\\submission_final.csv', 'valid': True, 'errors': [], 'shape': [500, 2], 'distribution': {0: 216, 1: 284}}`
- Hamming distance vs rollback baseline: `15`
- Recommendation: Submit `submissions/submission_final.csv`; stacker fallback selected the best single OOF model, so the final candidate stays conservative.

No test labels, LLM labels, pseudo-labeling, or public leaderboard tuning were used.
