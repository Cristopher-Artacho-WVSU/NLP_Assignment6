from collections import defaultdict
import math

class HMM:
    def __init__(self):
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.emission_counts = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)
        self.vocab = set()
        self.tags = set()
        self.total_sentences = 0

    def train(self, sentences):
        for sentence in sentences:
            self.total_sentences += 1
            words_tags = sentence.strip().split()
            prev_tag = 'START'

            for wt in words_tags:
                word, tag = wt.rsplit('_', 1)

                self.vocab.add(word)
                self.tags.add(tag)

                # Transition
                self.transition_counts[prev_tag][tag] += 1
                self.tag_counts[prev_tag] += 1

                # Emission
                self.emission_counts[tag][word] += 1
                self.tag_counts[tag] += 1

                prev_tag = tag

            # Transition to END
            self.transition_counts[prev_tag]['END'] += 1
            self.tag_counts[prev_tag] += 1

    def transition_prob(self, from_tag, to_tag):
        return self.transition_counts[from_tag][to_tag] / self.tag_counts[from_tag]

    def emission_prob(self, tag, word):
        return self.emission_counts[tag][word] / self.tag_counts[tag] if word in self.vocab else 1e-6

    def sequence_probability(self, tagged_sentence):
        """tagged_sentence: list of (word, tag) tuples"""
        prob = 1.0
        prev_tag = 'START'
        for word, tag in tagged_sentence:
            p_trans = self.transition_prob(prev_tag, tag)
            p_emit = self.emission_prob(tag, word)
            prob *= p_trans * p_emit
            prev_tag = tag
        prob *= self.transition_prob(prev_tag, 'END')
        return prob

    def log_sequence_probability(self, tagged_sentence):
        """To avoid underflow, return log probability"""
        log_prob = 0.0
        prev_tag = 'START'
        for word, tag in tagged_sentence:
            p_trans = self.transition_prob(prev_tag, tag)
            p_emit = self.emission_prob(tag, word)
            log_prob += math.log(p_trans + 1e-9) + math.log(p_emit + 1e-9)
            prev_tag = tag
        log_prob += math.log(self.transition_prob(prev_tag, 'END') + 1e-9)
        return log_prob

# === Example usage ===

# Training dataset
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

test_sentence = [('The', 'DET'), ('cat', 'NOUN'), ('meows', 'VERB')]
print("Test Sentence - Probability:", hmm.sequence_probability(test_sentence))