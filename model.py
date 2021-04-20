import transformers
import config
import torch.nn as nn

class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        x, _ = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        logits = self.l0(x)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits