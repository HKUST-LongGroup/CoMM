import nltk
from nltk.translate.meteor_score import meteor_score


def break_text_tokens(text_list):
    # nltk.download('punkt')
    # nltk.download('wordnet')
    tokens = [nltk.word_tokenize(text) for text in text_list]
    return tokens


def calculate_meteor(predict_text_steps, ground_truth_steps):
    predict_text_steps = break_text_tokens(predict_text_steps)
    ground_truth_steps = break_text_tokens(ground_truth_steps)
    ground_truth_steps = [[t] for t in ground_truth_steps]
    meteor_scores = [meteor_score(re, rs) for re, rs in zip(ground_truth_steps, predict_text_steps)]
    return meteor_scores, sum(meteor_scores) / len(meteor_scores)