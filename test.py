with open('/mnt/data/agedetector_group_train.v1.0.txt', 'r', encoding='utf-8') as f:
    train_data = f.readlines()

with open('/mnt/data/test.txt', 'r', encoding='utf-8') as f:
    test_data = f.readlines()
# Preprocess the data
def preprocess_data(data):
    labels = [line.split()[0][9:] for line in data]
    texts = [' '.join(line.split()[1:]) for line in data]
    return texts, labels

train_texts, train_labels = preprocess_data(train_data)
test_texts, test_labels = preprocess_data(test_data)

# Define the Naive Bayes classifier with Laplace smoothing
class NaiveBayesClassifier:
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.class_probs = defaultdict(float)
        self.word_probs = defaultdict(lambda: defaultdict(float))
        self.vocab = set()

    def fit(self, texts, labels):
        total_docs = len(texts)
        label_counts = Counter(labels)
        
        # Calculate prior probabilities
        for label, count in label_counts.items():
            self.class_probs[label] = count / total_docs
        
        # Calculate likelihood with Laplace smoothing
        for text, label in zip(texts, labels):
            words = text.split()
            word_counts = Counter(words)
            for word, count in word_counts.items():
                self.word_probs[label][word] += count
                self.vocab.add(word)
        
        # Normalize word probabilities
        V = len(self.vocab)
        for label, word_counts in self.word_probs.items():
            total_words = sum(word_counts.values())
            for word in self.vocab:
                self.word_probs[label][word] = (self.word_probs[label][word] + self.alpha) / (total_words + V * self.alpha)

    def predict(self, texts):
        predictions = []
        for text in texts:
            words = text.split()
            probs = defaultdict(float)
            for label, class_prob in self.class_probs.items():
                probs[label] = np.log(class_prob)  # Use log probability to prevent underflow
                for word in words:
                    if word in self.vocab:
                        probs[label] += np.log(self.word_probs[label][word])
            predictions.append(max(probs, key=probs.get))
        return predictions
    
# Preprocess the data
train_labels = [line.split()[0][9:] for line in train_data]
train_texts = [' '.join(line.split()[1:]) for line in train_data]

test_labels = [line.split()[0][9:] for line in test_data]
test_texts = [' '.join(line.split()[1:]) for line in test_data]
# Train the Naive Bayes model with Laplace smoothing
nb_custom = NaiveBayesClassifier(alpha=1)
nb_custom.fit(train_texts, train_labels)

# Predict on the test set
test_preds_custom = nb_custom.predict(test_texts)

# Evaluate the model
def evaluate_model(true_labels, predicted_labels):
    accuracy = sum(np.array(true_labels) == np.array(predicted_labels)) / len(true_labels)
    report = classification_report(true_labels, predicted_labels)
    return accuracy, report

accuracy_custom, classification_rep_custom = evaluate_model(test_labels, test_preds_custom)
accuracy_custom, classification_rep_custom
