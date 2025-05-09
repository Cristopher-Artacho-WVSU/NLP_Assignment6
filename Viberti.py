from collections import defaultdict
import math

class HMM:
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.emission_counts = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)
        self.vocab = set()
        self.tags = set()

    def train(self, sentences):
        for sentence in sentences:
            words_tags = sentence.strip().split()
            prev_tag = 'START'

            for wt in words_tags:
                word, tag = wt.rsplit('_', 1)
                self.vocab.add(word)
                self.tags.add(tag)

                self.transition_counts[prev_tag][tag] += 1
                self.tag_counts[prev_tag] += 1

                self.emission_counts[tag][word] += 1
                self.tag_counts[tag] += 1

                prev_tag = tag

            self.transition_counts[prev_tag]['END'] += 1
            self.tag_counts[prev_tag] += 1

    def transition_prob(self, prev_tag, curr_tag):
        return self.transition_counts[prev_tag][curr_tag] / self.tag_counts[prev_tag]

    def emission_prob(self, tag, word):
        # Apply add-one smoothing for unseen words
        return (self.emission_counts[tag][word] + 1) / (self.tag_counts[tag] + len(self.vocab))

    def viterbi(self, sentence):
        sentence = sentence.split()
        V = [{}]  # Viterbi table
        path = {}

        # Initialization
        for tag in self.tags:
            V[0][tag] = math.log(self.transition_prob('START', tag) + 1e-10) + \
                        math.log(self.emission_prob(tag, sentence[0]))
            path[tag] = [tag]

        # Recursion
        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for curr_tag in self.tags:
                (prob, prev_tag) = max(
                    (V[t - 1][pt] + math.log(self.transition_prob(pt, curr_tag) + 1e-10) + \
                     math.log(self.emission_prob(curr_tag, sentence[t])), pt)
                    for pt in self.tags
                )
                V[t][curr_tag] = prob
                new_path[curr_tag] = path[prev_tag] + [curr_tag]

            path = new_path

        # Termination
        n = len(sentence) - 1
        (prob, final_tag) = max((V[n][tag] + math.log(self.transition_prob(tag, 'END') + 1e-10), tag) for tag in self.tags)

        return path[final_tag], prob


# -----------------------------
# Training Dataset
# -----------------------------
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

# Initialize and train
hmm = HMM()
hmm.train(training_data)

# -----------------------------
# Test Sentences
# -----------------------------
test_sentence_a = "The can meows"
test_sentence_b = "My dog barks loudly"

tags, prob = hmm.viterbi(test_sentence_a)
print(f"Sentence: {test_sentence_a}")
print(f"Log Probability: {prob:.4f}")
print(f"Predicted Tags: {tags}")
print(f"Log Probability: {prob:.4f}")
print("-" * 40)

tags, prob = hmm.viterbi(test_sentence_b)
print(f"Sentence: {test_sentence_b}")
print(f"Log Probability: {prob:.4f}")
print(f"Predicted Tags: {tags}")
print(f"Log Probability: {prob:.4f}")
print("-" * 40)

