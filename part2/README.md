# Part 2: Diffusion Models

This part consists of four problems.
Each has its own notebook with more detailed instructions in the `part2` folder.

- Problem 1: Preliminaries and variance schedules
- Problem 2: Training diffusion models on a toy dataset
- Problem 3: MNIST and conditional generation
- Problem 4: Sampling from pre-trained models

## Submission Guideline for Part 2

Please include your answer to all problems, including formulas, proofs, and the figures generated in each problem, excluding code. You are required to submit the (single) PDF titled `part2.pdf` and all (four) notebooks with your code and running outputs. Do not include code in the PDF file. 

Specifically, the PDF should contain:
- Problem 1: Preliminaries and variance schedules
  - Formulas and proofs for problem 1.1 and 1.2
  - 4 figures, one for each beta schedule for problem 1.3
  - Answers to the 2 short answer questions about different beta schedules in problem 1.3
- Problem 2: Training diffusion models on a toy dataset
  - The generated figures `results/p2_train_plot.png` and `results/p2_toy_samples.png`
- Problem 3: MNIST and conditional generation
  - The generated figures `results/mnist_train_plot.png` and `results/image_w{w}.png` (w=0.0, 0.5, 1.0, 2.0, 4.0)
  - Answer to the short answer question about the U-Net architecture
  - Answer to the short answer question about different CFG weight $w$ in problem 3.2
- Problem 4: Sampling from pre-trained models
  - The generated figures `results/sampling_comparison.png
  - Answer to the short answer question about different samplers and number of steps
