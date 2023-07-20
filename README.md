# MIMIC-Diff-VQA
MIMIC-Diff-VQA is a large-scale dataset for visual question answering in medical chest x-ray images. This repository provides the code for generating MIMIC-Diff-VQA dataset from MIMIC-CXR database. For the code of the model used in our paper, "Expert Knowledge-Aware Image Difference Graph Representation Learning for Difference-Aware Medical Visual Question Answering," please refer to [EKAID](https://github.com/Holipori/EKAID).


## How to use
1. Enter the 'code' directory
    ```bash
    cd code
    ```
2. Prepare for the mimic_all.csv (MIMIC-CXR-JPG needs to be ready)
    ```bash
    python get_mimic_all.py -p <path_to_mimic_cxr_jpg>
    ```
3. Extract the intermediate KeyInfo json dataset
    ```bash
    python question_gen.py -j
    ```
4. Generate the full version of question answer pairs
    ```bash
    python question_gen.py -q
    ```

    Alternatively, you can execute step 3 and step 4 simultaneously by running:
    ```bash
    python question_gen.py -j -q
    ```
