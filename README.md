# Zero-shot (Multi-Label) Labelling with Roberta (RobertaForTokenZeroShotClassification)
This Roberta Model provides zero-shot labeling, i.e. predicting token-wise labels with only the sentence-level label.

The model is a gradient-based approach, with **max-min pooling** to learn localized features and **additional  loss functions** from Rei and Søgaard (2018) to encourage token-level classification.

## Requirements
* torch
* transformers

# Usage

The input requirements are the same as [RobertaForSequenceClassification](https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification), the model will only backpropagate the sentence-level loss, but the output will have both token-wise and sentence-wise label (i.e. token label in [RobertaForTokenClassification](https://huggingface.co/transformers/model_doc/roberta.html#robertafortokenclassification)). 

It supports **MultiLabel**!

The returned dictionary will contain both `logits` and `logit_mask`.
*   **`logits`**: Sentence level prediction, of shape (`batch_size`, `num_labels`).
*   **`logit_mask`**: Token level prediction, of shape (`batch_size`, `sequence_length`, `num_labels`).


## Hyperparameters Explanation
*   **`num_maps`**: Numbers of intermediate feature maps per class through 1x1 convolutions.
*   **`kmax`**: Percentage of maximum to included and propagated.
*   **`kmin`**: Percentage of minimum to included and propagated.
*   **`alpha`**: Weight of minimum features in the objective function.
*   **`beta`**: Power of weighted attention, larger `beta` means more focus prediction.
*   **`penalty_ratio`**: Ratio of extra loss. (extra loss introduced in Rei and Søgaard 2018)
*   **`random_drop`**: Ratio of token's random drop during training.


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
| Transformer (Bujel, Yannakoudakis, and Rei) | 20.76  | 85.36  | 33.31  |
| Roberta Zero-Shot             | 25.47  | 63.16  | **36.30**  |
