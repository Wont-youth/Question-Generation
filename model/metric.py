from nltk.translate.bleu_score import sentence_bleu
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from nltk.translate import meteor_score
from rouge import Rouge

def calculate_bleu(reference, candidate):
    bleu_1_gram = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu_2_gram = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu_4_gram = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu_1_gram, bleu_2_gram, bleu_4_gram

def calculate_meteor(reference, candidate):
    meteor = meteor_score.meteor_score(reference, candidate)
    return meteor

def calculate_rough(reference, candidate):
    rouge = Rouge()
    rough_score = rouge.get_scores(candidate, reference)
    rough_l = rough_score[0]["rouge-l"]
    rough_l_r = rough_l["r"]
    return rough_l_r