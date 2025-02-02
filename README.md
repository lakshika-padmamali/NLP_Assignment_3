# NLP_Assignment_3
# **Text Simplifier and English-Sinhala Translator**

## Overview

This repository contains an implementation of an **English Text Simplifier** and an additional project **English-Sinhala Translator**:

1. **Text Simplifier**: A model trained to simplify English sentences using General, Multiplicative, and Additive Attention mechanisms.
2. **English-Sinhala Translator**: A parallel english and sinhala word translator-based English to Sinhala translation model.

## **Text Simplifier**

## Project Structure

- **Attention Mechanisms**: Implemented General, Multiplicative, and Additive Attention for text simplification.

- **Dataset Sources**:

  - The dataset for text simplification is derived from **Simple English Wikipedia** and **English Wikipedia**. This dataset was created by aligning sentences from both versions of Wikipedia.
  - **Version 1.0** (May 2010): Contains **137K aligned sentence pairs**, filtered using **TF-IDF similarity thresholding**.
  - **Version 2.0** (May 2011): Contains **167K aligned sentence pairs**, with improved text processing and updated Wikipedia data. Used in *Improving Text Simplification Language Modeling Using Unsimplified Text Data* (David Kauchak, 2013, ACL Proceedings).
  - Further details about this dataset can be found in: *Simple English Wikipedia: A New Simplification Task* (William Coster & David Kauchak, 2011, ACL Proceedings).

- **Evaluation and Metrics**: Training loss, validation loss, perplexity (PPL), and attention maps.

- **Performance Comparisons**: Training results for different attention mechanisms.

## Training Results

### General Attention

| Epoch                              | Train Loss | Train PPL | Val Loss | Val PPL |
| ---------------------------------- | ---------- | --------- | -------- | ------- |
| 1                                  | 5.196      | 180.542   | 4.464    | 86.815  |
| 2                                  | 4.154      | 63.692    | 3.970    | 52.963  |
| 3                                  | 3.656      | 38.698    | 3.708    | 40.777  |
| 4                                  | 3.329      | 27.912    | 3.549    | 34.778  |
| 5                                  | 3.096      | 22.100    | 3.452    | 31.563  |
| 6                                  | 2.925      | 18.626    | 3.383    | 29.464  |
| 7                                  | 2.794      | 16.352    | 3.341    | 28.246  |
| 8                                  | 2.693      | 14.781    | 3.308    | 27.323  |
| 9                                  | 2.610      | 13.598    | 3.277    | 26.498  |
| 10                                 | 2.541      | 12.696    | 3.259    | 26.015  |
| **Training Time:** 3050.46 seconds |            |           |          |         |

### Multiplicative Attention

| Epoch                              | Train Loss | Train PPL | Val Loss | Val PPL |
| ---------------------------------- | ---------- | --------- | -------- | ------- |
| 1                                  | 5.193      | 179.972   | 4.471    | 87.436  |
| 2                                  | 4.158      | 63.913    | 3.980    | 53.500  |
| 3                                  | 3.658      | 38.800    | 3.709    | 40.829  |
| 4                                  | 3.330      | 27.943    | 3.560    | 35.174  |
| 5                                  | 3.098      | 22.145    | 3.460    | 31.830  |
| 6                                  | 2.926      | 18.656    | 3.396    | 29.847  |
| 7                                  | 2.795      | 16.358    | 3.350    | 28.506  |
| 8                                  | 2.693      | 14.769    | 3.314    | 27.486  |
| 9                                  | 2.610      | 13.601    | 3.281    | 26.610  |
| 10                                 | 2.543      | 12.719    | 3.260    | 26.052  |
| **Training Time:** 3069.24 seconds |            |           |          |         |

### Additive Attention

| Epoch                              | Train Loss | Train PPL | Val Loss | Val PPL |
| ---------------------------------- | ---------- | --------- | -------- | ------- |
| 1                                  | 5.209      | 182.961   | 4.480    | 88.240  |
| 2                                  | 4.167      | 64.534    | 3.975    | 53.251  |
| 3                                  | 3.662      | 38.938    | 3.706    | 40.678  |
| 4                                  | 3.328      | 27.896    | 3.558    | 35.097  |
| 5                                  | 3.096      | 22.114    | 3.458    | 31.748  |
| **Training Time:** 3045.51 seconds |            |           |          |         |

## Evaluation and Verification

1. **Comparison of Attention Mechanisms:**
   
General, Multiplicative, and Additive Attention mechanisms were compared based on translation accuracy and computational efficiency.
General Attention achieved the best trade-off between computational efficiency and translation accuracy.

Multiplicative Attention had a slight improvement in perplexity (PPL) over General Attention but required higher computational power.

Additive Attention resulted in similar translation accuracy but was slightly less efficient in training time compared to General Attention.

Final Model Selection: Based on overall performance, General Attention was selected as the best model and saved for further deployment.

2. **Performance Plots:**

   
   -![attention general](https://github.com/user-attachments/assets/1b02abcc-13d6-4521-8100-262510f66eee)
   ![attention multiplicative](https://github.com/user-attachments/assets/4006b651-776d-4946-9926-dd2db09a1ce5)
   ![training_loss_comparison](https://github.com/user-attachments/assets/22c19999-654c-45df-b72b-82789fdfadb2)



4. **Attention Maps:** Visual representations of model attention to different words during translation.
5. **Effectiveness Analysis:** The results suggest that **General Attention** provides a good balance between computational efficiency and translation accuracy, making it the best candidate for real-world text simplification tasks.

## **Additional Project: English-Sinhala Translator**

Implemented an **English-Sinhala parallel word translator** for bilingual translation.

### **Dataset Credits**

- **Sinhala-English Dictionary**: Extracted from [Sinhala-Para-Dict](https://github.com/kasunw22/sinhala-para-dict/blob/main/README.md), based on:
  - **@INPROCEEDINGS{Wick2308****:Sinhala****, IEEE ICIIS 2023**: *Sinhala-English Parallel Word Dictionary Dataset* by Kasun Wickramasinghe and Nisansa de Silva.

## Project Demonstrations

### **Text Simplifier UI**
![text simplifier](https://github.com/user-attachments/assets/c58b7702-c698-4dbb-b4b7-1ae8ba7410d6)



### **English-Sinhala Translator UI**

![sinhala english translator](https://github.com/user-attachments/assets/b60ec810-d9da-42d0-8179-26fdba09cb46)




