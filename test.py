# 1. Định nghĩa các hàm và lớp cần thiết

# Preprocess the data
def preprocess_data(data):
    labels = [line.split()[0][9:] for line in data]
    texts = [' '.join(line.split()[1:]) for line in data]
    return texts, labels

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

# Custom evaluation metrics
def accuracy(true_labels, predicted_labels):
    correct_predictions = sum(a == b for a, b in zip(true_labels, predicted_labels))
    return correct_predictions / len(true_labels)

def classification_metrics(true_labels, predicted_labels):
    labels = list(set(true_labels))
    metrics = {}
    
    for label in labels:
        tp = sum(y_true == label and y_pred == label for y_true, y_pred in zip(true_labels, predicted_labels))
        fp = sum(y_true != label and y_pred == label for y_true, y_pred in zip(true_labels, predicted_labels))
        fn = sum(y_true == label and y_pred != label for y_true, y_pred in zip(true_labels, predicted_labels))
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        metrics[label] = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        }
        
    return metrics

# 2. Tải và tiền xử lý dữ liệu
with open('/mnt/data/agedetector_group_train.v1.0.txt', 'r', encoding='utf-8') as f:
    train_data = f.readlines()
with open('/mnt/data/test.txt', 'r', encoding='utf-8') as f:
    test_data = f.readlines()

train_texts, train_labels = preprocess_data(train_data)
test_texts, test_labels = preprocess_data(test_data)

# 3. Huấn luyện và dự đoán sử dụng phân loại Naive Bayes
nb_custom = NaiveBayesClassifier(alpha=1)
nb_custom.fit(train_texts, train_labels)
test_preds_custom = nb_custom.predict(test_texts)

# 4. Đánh giá kết quả
accuracy_val = accuracy(test_labels, test_preds_custom)
classification_metrics_val = classification_metrics(test_labels, test_preds_custom)

accuracy_val, classification_metrics_val
