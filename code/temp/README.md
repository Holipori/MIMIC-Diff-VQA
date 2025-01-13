# Fix Study Order issue

## Problem
In the dataset, the reference study is supposed to precede the current study for "difference" questions. However, there are some questions are not following this rule due to a bug during generation.

## Solution
The original dataset has consistent question and answer pairs. The answers are all correct in the current setting. If you have trained your model on the original dataset, this issue should not affect the performance of your model. The only issue is the order of the study_id and ref_id for some of the questions.

We have fixed the bug in the code for generating the dataset.

We also uploaded an updated version of the dataset to Physionet and currently under review. The new dataset has minimum changes, achieved by correcting the order of the problematic questions and updating the answers accordingly.

Before the new dataset is available, we temporarily provide the code for fixing this issue. Please follow the steps below:
1. Download the dataset from [Physionet](https://physionet.org/content/medical-diff-vqa/1.0.0/) and put the files into the project folder './'
2. Run the following code at `./code/temp` folder:
```bash
python fix_study_order.py
```
3. The fixed dataset will be saved in `./code/temp` folder.