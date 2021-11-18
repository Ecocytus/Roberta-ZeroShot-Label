# Zero-shot Labelling with Roberta (RobertaForTokenZeroShotClassification)
This Roberta Model provides zero-shot labeling, i.e. predicting token-wise labels with only sentence-level label.

## Requirements
* torch
* transformers

# Usage

The input requirements are the same as [RobertaForSequenceClassification](https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification). (It supports **multilabel**!)

The returned dictionary will contain both `logits` and `logit_mask` (the same as [RobertaForTokenClassification](https://huggingface.co/transformers/model_doc/roberta.html#robertafortokenclassification)).
*   **`logits`**: Sentence level prediction, of shape (`batch_size`, `num_labels`).
*   **`logit_mask`**: Token level prediction, of shape (`batch_size`, `sequence_length`, `num_labels`).


## Hyperparameters Explanation
*   **`num_maps`**: Numbers of intermediate feature maps per class through 1x1 convolutions.
*   **`kmax`**: Pencentage of maximum to included and propergated.
*   **`kmin`**: Pencentage of minimum to included and propergated.
*   **`alpha`**: Weight of minimum features in the objective funtion.
*   **`beta`**: Power of weighted attetion, larger `beta` means more focus prediction.
*   **`penalty_ratio`**: Ratio of extra loss. (extra loss introduced in Rei and Søgaard 2018)
*   **`random_drop`**: Ratio of token's random drop during traning.


## Data
Currently only trained with First Certificate in English (FCE) dataset, the preprocessed FCE dataset is included in the repo.

## Train
Details in the FCE_train.ipynb.

## Performance
Compared to other zero-shot labeling models on FCE dataset.

|                               | FCE    |        |        |
|-------------------------------|--------|--------|--------|
|                               | P      | R      | F1     |
| LIME                          | 19.06  | 34.70  | 24.60  |
| LSTM (Rei and Søgaard)        | 29.16  | 29.04  | 28.73  |
| Transformer (Rei and Søgaard) | 20.76  | 85.36  | 33.31  |
| Roberta Zero-Shot             | 25.47  | 63.16  | **36.30**  |
