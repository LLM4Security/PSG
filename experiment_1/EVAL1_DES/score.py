# %%
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu,SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge import Rouge 
import pandas as pd

# %%
def compute_sentence_bleu(reference, hypothesis):
    # reference = [reference.split()]
    # hypothesis = hypothesis.split()
    reference = [word_tokenize(reference)]
    hypothesis = word_tokenize(hypothesis)
    weights = [(1.0, 0.0, 0.0, 0.0),  # BLEU-1
               (0.5, 0.5, 0.0, 0.0),  # BLEU-2
               (0.33, 0.33, 0.33, 0.0),  # BLEU-3
               (0.25, 0.25, 0.25, 0.25)]  # BLEU-4
    smoothie = SmoothingFunction().method1
    bleu_scores = []
    for n in range(1, 5):
        bleu = sentence_bleu(reference, hypothesis, weights[n-1], smoothing_function = smoothie)
        bleu_scores.append(bleu * 100)

    return bleu_scores

def compute_corpus_bleu(reference, hypothesis):
    reference = [reference]
    hypothesis = [hypothesis]
    # reference = [word_tokenize(reference)]
    # hypothesis = word_tokenize(hypothesis)
    weights = [(1.0, 0.0, 0.0, 0.0),  # BLEU-1
               (0.5, 0.5, 0.0, 0.0),  # BLEU-2
               (0.33, 0.33, 0.33, 0.0),  # BLEU-3
               (0.25, 0.25, 0.25, 0.25)]  # BLEU-4
    bleu_scores = []
    for n in range(1, 5):
        bleu = corpus_bleu(reference, hypothesis, weights[n-1])
        bleu_scores.append(bleu * 100)

    return bleu_scores

def compute_rouge_l(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores([hypothesis], [reference])
    rouge_l_score = scores[0]['rouge-l']['f']
    
    return rouge_l_score * 100
# %%
bleu_scores = compute_sentence_bleu('Tap to take a photo'.lower(), 'the ui includes options to take a photo with the camera'.lower())
print(sum(bleu_scores)/len(bleu_scores))
# %%
result = './data.csv'
# result = './results_ctx.csv'


df = pd.read_csv(result)

# %%


for index, row in df.iterrows():
    reference_sentence = row['des']
    hypothesis_sentence = row['des_psg']
    # hypothesis_sentence = row['des_ctx']

    
    bleu_scores = compute_sentence_bleu(reference_sentence.lower(), hypothesis_sentence.lower())
    # bleu_scores = compute_corpus_bleu(reference_sentence.lower(), hypothesis_sentence.lower())
    rouge_l_score = compute_rouge_l(reference_sentence.lower(), hypothesis_sentence.lower())
    # df.at[index, 'BLEU-1'] = bleu_scores[0]
    # df.at[index, 'BLEU-2'] = bleu_scores[1]
    # df.at[index, 'BLEU-3'] = bleu_scores[2]
    # df.at[index, 'BLEU-4'] = bleu_scores[3]
    df.at[index, 'BLEU-n'] = sum(bleu_scores)/len(bleu_scores)
    df.at[index, 'ROUGE-L'] = rouge_l_score

# print(compute_corpus_bleu(list(df['Reference']), list(df['Hypothesis'])))
# %%
# df = df[df["BLEU-n"] > 0]
# df = df[df["BLEU-n"] < 100]
# df = df[df["ROUGE-L"] > 0]
# df = df[df["ROUGE-L"] < 100]

print('BLEU-n: {:.2f} ± {:.2f}'.format(df["BLEU-n"].mean(), df["BLEU-n"].std()))
print('ROUGE-L: {:.2f} ± {:.2f}'.format(df["ROUGE-L"].mean(), df["ROUGE-L"].std()))

# %%
print(df.groupby('per')['BLEU-n'].mean())
print(df.groupby('per')['BLEU-n'].std())
print(df.groupby('per')['ROUGE-L'].mean())
print(df.groupby('per')['ROUGE-L'].std())
# %%
# df.sort_values('BLEU-n', ascending=False)
# 保存修改后的DataFrame到CSV文件
# df['per'] = df['per'].apply(lambda x: x.lower())
# df.to_csv(result, index=False)

# %%