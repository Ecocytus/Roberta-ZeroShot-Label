from torch.nn import BCEWithLogitsLoss, BCELoss
import torch
from torch import nn
from transformers import RobertaModel
from pooler import ClassWisePool, MaxMinPool1d

# code is developed on Roberta implementation and WildCatPooling(traditional maxmin pooling)
class RobertaForTokenZeroShotClassification(nn.Module):
    def __init__(self, roberta, num_labels, num_maps, kmax=1, kmin=0, alpha=1, beta=2, penalty_ratio=0.01, random_drop=0.1):
        super().__init__()
        if roberta is not None:
            self.roberta = roberta
        else:
            self.roberta = RobertaModel.from_pretrained('roberta-base', add_pooling_layer=False)
        config = self.roberta.config
        self.config = config
        self.embed_dim = 768
        self.num_labels = num_labels
        self.penalty_ratio = penalty_ratio
        self.random_drop = random_drop

        self.dropout = nn.Dropout(p=0.1)

        self.num_maps = num_maps

        self.classifier = nn.Conv1d(in_channels=self.embed_dim,
                      out_channels=self.num_labels*num_maps,
                      kernel_size=1)

        self.class_wise_pooling = ClassWisePool(num_maps)

        self.spatial_pooling = MaxMinPool1d(kmax, kmin, alpha, beta, random_drop)

    def update_dropout(self, p=0.5):
        self.dropout = nn.Dropout(p)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
        return_convs=False,
        return_mask=False
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        x_embed = outputs[0]

        # not use CLS token and EOS token
        mask = torch.bitwise_and((input_ids != 2).type(attention_mask.dtype).to(attention_mask.device), attention_mask).to(attention_mask.device)

        x_embed = (x_embed*mask.unsqueeze(-1))[:, 1:, :]
        
        x_embed = x_embed.permute(0, 2, 1)

        x_embed = self.dropout(x_embed)

        mask = self.classifier(x_embed)

        average_mask = self.class_wise_pooling(mask)

        logits, extra_loss = self.spatial_pooling(average_mask, labels)

        loss_fct = BCEWithLogitsLoss()

        if labels is not None:
            loss = loss_fct(logits, labels) + extra_loss*self.penalty_ratio
        else:
            loss = None

        result = {"logits": logits, "loss": loss}

        if return_mask:
            result['raw_mask'] = mask
            result['average_mask'] = average_mask

        return result