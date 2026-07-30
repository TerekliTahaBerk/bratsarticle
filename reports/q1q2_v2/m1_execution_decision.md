# M1 execution decision

Decision: **LOSS SCREEN FEASIBLE; FULL MATRIX BLOCKED**

The restart-safe loss screen contains 15 jobs and is estimated at 9.8-16.4 serial hours on the measured M1 Max.

The 10 measured convergence models alone require 2,132.2 optimizer-work hours at the frozen 50,000-step ceiling. Adding the fixed core compute regime, the known four interaction finalists, and the loss screen yields 3,512.5 hours (146.4 serial days). This is not a complete total: both nnU-Net baselines, repeated validation, full metrics, reproduction, and external inference are excluded.

On fold 1, fast selection validation took 98.6 seconds versus 189.6 seconds for the full evaluator and matched mean regional Dice exactly.

Therefore only the bounded development loss screen may start now. Five folds and five common main seeds remain unchanged. Gate F and external inference remain blocked.
