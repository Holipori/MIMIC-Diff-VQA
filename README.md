# Medical-Diff-VQA
Medical-Diff-VQA (originally MIMIC-Diff-VQA) is a large-scale dataset for visual question answering in medical chest x-ray images. This repository provides the code for generating Medical-Diff-VQA dataset, which is proposed in our paper "**Expert Knowledge-Aware Image Difference Graph Representation Learning for Difference-Aware Medical Visual Question Answering**" 

For more information about the dataset and the method, please refer to our [paper](https://dl.acm.org/doi/abs/10.1145/3580305.3599819) or [project page](https://holipori.github.io/KDD2023-MIMIC-Diff-VQA/).

For the code of the method, please refer to [EKAID](https://github.com/Holipori/EKAID).

The Medical-Diff-VQA dataset is now available on [Physionet](https://physionet.org/content/medical-diff-vqa/1.0.0/).



[//]: # (We will provide the download link as soon as it is available. Since our dataset contains sensitive data, there are some necessary procedures to complete before accessing it. We suggest completing these procedures while waiting, so that you can save time once our dataset is finally released. Firstly, you need to apply to become a credentialed user on Physionet. After that, you need to complete the CITI Data or Specimens Only Research training. For more information, please refer to [this page]&#40;https://physionet.org/settings/credentialing/&#41;. )

## How to use
To generate a new Medical-Diff-VQA dataset(due to the randomness, the generated dataset will not be 100% the same as our provided one), please follow the steps below:
1. Enter the 'code' directory
    ```bash
    cd code
    ```
2. Prepare for the mimic_all.csv (MIMIC-CXR-JPG needs to be ready)
    ```bash
    python get_mimic_all.py -p <path_to_mimic_cxr_jpg>
    ```
3. Extract the intermediate KeyInfo json dataset. The <path_to_reports_folder> refers to the "files" folder that is unzipped from the mimic-cxr-reports.zip file in the MIMIC-CXR database.
    ```bash
    python question_gen.py -j -r <path_to_reports_folder>
    ```
4. Generate the full version of question answer pairs
    ```bash
    python question_gen.py -q
    ```

    Alternatively, you can execute step 3 and step 4 simultaneously by running:
    ```bash
    python question_gen.py -j -q
    ```
