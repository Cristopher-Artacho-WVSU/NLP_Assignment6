from collections import defaultdict
import math

# Training data
def train_hmm(sentences):
    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)
    vocab = set()
    tags = set()

    for sentence in sentences:
        words_tags = sentence.strip().split()
        prev_tag = "START"

        for wt in words_tags:
            word, tag = wt.rsplit('_', 1)
            vocab.add(word)
            tags.add(tag)

            # Transition count
            transition_counts[prev_tag][tag] += 1
            tag_counts[prev_tag] += 1

            # Emission count
            emission_counts[tag][word] += 1
            tag_counts[tag] += 1

            prev_tag = tag

        # Final END transition
        transition_counts[prev_tag]["END"] += 1
        tag_counts[prev_tag] += 1

    return transition_counts, emission_counts, tag_counts, vocab, tags

def transition_prob(transition_counts, tag_counts, from_tag, to_tag):
    # Add-one smoothing
    return (transition_counts[from_tag][to_tag] + 1) / (tag_counts[from_tag] + len(tag_counts))

def emission_prob(emission_counts, tag_counts, tag, word, vocab):
    # Add-one smoothing for unknown words
    if word not in vocab:
        return 1 / (tag_counts[tag] + len(vocab))
    return emission_counts[tag][word] / tag_counts[tag]

def viterbi(sentence, transition_counts, emission_counts, tag_counts, vocab, tags):
    words = sentence.split()
    V = [{}]  # Viterbi table
    path = {}

    # Initialization
    for tag in tags:
        tp = transition_prob(transition_counts, tag_counts, "START", tag)
        ep = emission_prob(emission_counts, tag_counts, tag, words[0], vocab)
        V[0][tag] = math.log(tp) + math.log(ep)
        path[tag] = [tag]

    # Recursion
    for t in range(1, len(words)):
        V.append({})
        new_path = {}

        for curr_tag in tags:
            max_prob, prev_state = max(
                ((V[t - 1][prev_tag] +
                  math.log(transition_prob(transition_counts, tag_counts, prev_tag, curr_tag)) +
                  math.log(emission_prob(emission_counts, tag_counts, curr_tag, words[t], vocab)),
                  prev_tag)
                 for prev_tag in tags),
                key=lambda x: x[0]
            )
            V[t][curr_tag] = max_prob
            new_path[curr_tag] = path[prev_state] + [curr_tag]

        path = new_path

    # Termination
    max_prob, final_tag = max(
        ((V[len(words) - 1][tag] + math.log(transition_prob(transition_counts, tag_counts, tag, "END")), tag)
         for tag in tags),
        key=lambda x: x[0]
    )

    return path[final_tag], max_prob
if __name__ == "__main__":
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

    # Train
    trans, emit, counts, vocab, tags = train_hmm(training_data)

    # Test sentences
    test_a = "The can meows"
    test_b = "My dog barks loudly"

    # Run Viterbi
    tags_a, prob_a = viterbi(test_a, trans, emit, counts, vocab, tags)
    tags_b, prob_b = viterbi(test_b, trans, emit, counts, vocab, tags)

    print(f"Sentence A: {test_a}")
    print(f"Predicted Tags: {tags_a}, Log Probability: {prob_a:.4f}\n")

    print(f"Sentence B: {test_b}")
    print(f"Predicted Tags: {tags_b}, Log Probability: {prob_b:.4f}")