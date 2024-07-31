import nltk
from nltk.translate.bleu_score import sentence_bleu


def break_text_tokens(text_list):
    nltk.download('punkt')
    tokens = [nltk.word_tokenize(text) for text in text_list]
    return tokens

# Calculate BLEU score
def calculate_bleu(predict_text_steps, ground_truth_steps):
    predict_text_steps = break_text_tokens(predict_text_steps)
    ground_truth_steps = break_text_tokens(ground_truth_steps)
    ground_truth_steps = [[t] for t in ground_truth_steps]
    bleu_scores = [sentence_bleu(re, rs) for re, rs in zip(ground_truth_steps, predict_text_steps)]
    return bleu_scores, sum(bleu_scores) / len(bleu_scores)



