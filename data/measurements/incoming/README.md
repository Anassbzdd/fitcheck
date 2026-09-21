# Imported Kaggle validation archive

`final-t4-20260921-133520.zip` is an imported Kaggle T4 validation archive. It is kept
outside the top-level measurement manifest until its rows are converted to the canonical
`runs` format and declared explicitly.

Summary:

- 24 registered rows: 12 accuracy rows, 6 predicted-safe-boundary rows, and 6 rows above the predicted safety boundary.
- 12 successful accuracy rows; 4 expected physical OOM rows; the remaining boundary rows completed above FitCheck's 14,000 MiB T4 safety budget.
- Stack: Tesla T4 (`sm_75`), torch `2.10.0+cu128`, transformers `5.0.0`, peft `0.19.1`, bitsandbytes `0.50.2`.
- Accuracy: mean absolute process error `5.4%`; worst under-prediction `-15.3%`; tensor-tier errors stayed within `±1.5%`.
- The Qwen 1.5B Coder rows expose a process-overhead miss; this archive must not be used to fit shipped coefficients without a canonical conversion and manifest review.
