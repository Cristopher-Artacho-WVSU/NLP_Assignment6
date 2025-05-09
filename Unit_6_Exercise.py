from collections import defaultdict
import math

def train_hmm(sentences):
    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)
    vocab = set()
    tags = set()

    for sentence in sentences:
        prev_tag = "START"
        words_tags = sentence.strip().split()

        for wt in words_tags:
            word, tag = wt.rsplit('_', 1)
            vocab.add(word)
            tags.add(tag)

            
            transition_counts[prev_tag][tag] += 1
            tag_counts[prev_tag] += 1

            
            emission_counts[tag][word] += 1
            tag_counts[tag] += 1

            prev_tag = tag

        transition_counts[prev_tag]["END"] += 1
        tag_counts[prev_tag] += 1

    return transition_counts, emission_counts, tag_counts, vocab, tags

def transition_prob(transition_counts, tag_counts, from_tag, to_tag):
    return (transition_counts[from_tag][to_tag] + 1) / (tag_counts[from_tag] + len(transition_counts[from_tag]))

def emission_prob(emission_counts, tag_counts, tag, word, vocab):
    if word in vocab:
        return (emission_counts[tag][word] + 1) / (tag_counts[tag] + len(emission_counts[tag]))
    else: 
        return 1 / (tag_counts[tag] + len(emission_counts[tag]))

def viterbi(sentence, transition_counts, emission_counts, tag_counts, vocab, tags):
    V = [{}]
    path = {}

    for tag in tags:
        V[0][tag] = math.log(transition_prob(transition_counts, tag_counts, "START", tag)) + \
                    math.log(emission_prob(emission_counts, tag_counts, tag, sentence[0], vocab))
        path[tag] = [tag]

    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}

        for curr_tag in tags:
            (prob, prev_tag) = max(
                (V[t-1][ptag] + math.log(transition_prob(transition_counts, tag_counts, ptag, curr_tag)) +
                 math.log(emission_prob(emission_counts, tag_counts, curr_tag, sentence[t], vocab)), ptag)
                for ptag in tags
            )
            V[t][curr_tag] = prob
            new_path[curr_tag] = path[prev_tag] + [curr_tag]

        path = new_path

    (prob, final_tag) = max(
        (V[len(sentence) - 1][tag] + math.log(transition_prob(transition_counts, tag_counts, tag, "END")), tag)
        for tag in tags
    )
    return path[final_tag]

training_data = [
    "The_DET cat_NOUN sleeps_VERB",
    "A_DET dog_NOUN barks_VERB",
    "The_DET dog_NOUN sleeps_VERB",
    "My_DET dog_NOUN runs_VERB fast_ADV",
    "A_DET cat_NOUN meows_VERB loudly_ADV",
    "Your_DET cat_NOUN runs_VERB",
    "The_DET bird_NOUN sings_VERB sweetly_ADV",
    "A_DET bird_NOUN chirps_VERB"
]

transition_counts, emission_counts, tag_counts, vocab, tags = train_hmm(training_data)

test_a = ["The", "can", "meows"]
test_b = ["My", "dog", "barks", "loudly"]

result_a = viterbi(test_a, transition_counts, emission_counts, tag_counts, vocab, tags)
result_b = viterbi(test_b, transition_counts, emission_counts, tag_counts, vocab, tags)

print("Sentence A:", test_a)
print("Best POS Sequence:", result_a)

print("\nSentence B:", test_b)
print("Best POS Sequence:", result_b)


