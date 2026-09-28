# GPU-free completion checklist

User requested all eight preparation items on 2026-09-29. No rented GPU or
pretrained GPU execution is authorized by this checklist.

| Item | Status | Completion evidence required |
| --- | --- | --- |
| 1. Genuine LIBERO fixtures | Complete locally | Three real spatial-task starts, seed 7, raw/processed images and native states; local OpenGL |
| 2. A1.5/W0 checkpoint audit | Complete | Both full published hashes match; 950 A1.5 tensors, 1660 W0 tensors; 824 W0 motor shapes match native architecture |
| 3. Real-data batching | CPU complete | 50 complete windows, 25 episodes, 13 reloadable batches with explicit provenance/masks |
| 4. Broader FAST validation | CPU complete | 50 windows, 19-73 codes, mean active RMSE 0.01197; malformed-label tests |
| 5. Vocabulary expansion | CPU complete | Tied/untied tests plus actual tiny Qwen class, exact weight resume and optimizer reload |
| 6. Donor transfer adapters | CPU contracts complete | Real native reduced W0 and patched A1.5 forward/backward; W0 preconditioned K/V load corrected and exact roundtrip checked |
| 7. Evaluation protocols | Complete | See EVALUATION_PROTOCOL.md; native nine-episode subset, research budgets and reporting gates |
| 8. GPU bundles | In progress | Native motor numerical capture command prepared; awaiting real capture/checkpoint inventory |

Prepared commands or toy tests are not pretrained-policy qualification. Missing
hardware-dependent evidence must remain explicitly pending, even after all
GPU-free preparation is done.
