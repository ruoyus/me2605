# Variable-Length Attention Lab

Open the published page to use **Train**, **Pause**, and **Reset**. Each click on **Train** runs the indicated number of additional updates. Both models learn from the same batches of 32 freshly sampled training prompts in a Web Worker. Reset returns to seeded random weights and step zero. These browser runs are separate from the NumPy benchmark.

The saved model and test metrics in `results.js` also work when `index.html` is opened as a local file. For live training on your computer, run `python3 -m http.server 8000` in this folder and open `http://localhost:8000/`.

Run `python3 train.py --seed 7 --steps 1600` with NumPy to regenerate the saved data. Both models use the same seeded corpus and mini-batch sequence. The corpus has 20 fruits, 30 vegetables, 24 distractors, and one of each ingredient per prompt. It reserves 120 ingredient pairs for test and uses the remaining 480 for training. Each pair contributes both category queries at each length from 7 to 30.

The last real token is the attention query. Padding is excluded from the attention sum. The MLP concatenates 30 embeddings (zero vectors for padding) and uses a width-11 hidden layer. Its input width is 480. The web interface rejects prompts over 30 and leaves the submitted text in place. The fixed-width restriction is part of the MLP architecture. For attention it is a demo and evaluation choice.

This is one empirical comparison. The architectures have different parameter counts. The test results measure held-out ingredient combinations within the tested length range.
