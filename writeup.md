
Pretrained Word + LSTM : 0.42
Pretrained Word + ByteLSTM + LSTM : 0.4818
GLoVE WE (tunable) + ByteLSTM + dropout + LSTM : 0.59

Embeds: Done
- Glove
- fast text

Recurrent unit:
- BiLSTM
- BiGRU

RU layers: 0-5

CRF: Doesn't work, data too unbiased to make a difference
- Rule based
- Viterbi
