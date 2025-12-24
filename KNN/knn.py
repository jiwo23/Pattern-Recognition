
import os
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time

# Regex to parse reviews
# Handles: <review id="123">...</review> and <review id="123" label="1">...</review>
REVIEW_PATTERN = re.compile(r'<review id="(?P<id>\d+)"(?:\s+label="(?P<label>\d+)")?>(?P<text>.*?)</review>', re.DOTALL)

def load_data(file_path, default_label=None):
    texts = []
    labels = []
    
    print(f"Reading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Fallback for gbk if utf-8 fails (common in chinese datasets)
        with open(file_path, 'r', encoding='gb18030') as f:
            content = f.read()

    matches = REVIEW_PATTERN.finditer(content)
    for match in matches:
        text = match.group('text').strip()
        label_str = match.group('label')
        
        if label_str is not None:
            labels.append(int(label_str))
        elif default_label is not None:
            labels.append(default_label)
        else:
            # If no label found and no default provided, skip or handle as needed
            # For this task, training data needs default_label
            continue
            
        texts.append(text)
        
    print(f"Loaded {len(texts)} samples from {file_path}")
    return texts, labels

def tokenize(text):
    return jieba.lcut(text)

def main():
    start_time = time.time()
    
    # Paths
    train_pos_path = r'train/evaltask2_sample_data/cn_sample_data/sample.positive.txt'
    train_neg_path = r'train/evaltask2_sample_data/cn_sample_data/sample.negative.txt'
    test_label_path = r'test mark/Sentiment Classification with Deep Learning/test.label.cn.txt'
    
    # Load Training Data
    pos_texts, pos_labels = load_data(train_pos_path, default_label=1)
    neg_texts, neg_labels = load_data(train_neg_path, default_label=0)
    
    train_texts = pos_texts + neg_texts
    train_labels = pos_labels + neg_labels
    
    # Load Test Data
    test_texts, test_labels = load_data(test_label_path)
    
    print(f"\nTotal Training Samples: {len(train_texts)}")
    print(f"Total Test Samples: {len(test_texts)}")
    
    # Vectorization
    print("\nVectorizing text...")
    # limiting features to avoid memory issues and reduce noise
    vectorizer = TfidfVectorizer(tokenizer=tokenize, max_features=5000) 
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"Feature matrix shape: {X_train.shape}")
    
    # KNN Classifier
    print("\nTraining KNN Classifier (k=10, metric='cosine')...")
    knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
    knn.fit(X_train, train_labels)
    
    # Prediction
    print("Predicting on test set...")
    y_pred = knn.predict(X_test)
    
    # Evaluation
    print("\n" + "="*40)
    print("SENTIMENT ANALYSIS REPORT")
    print("="*40)
    
    acc = accuracy_score(test_labels, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, target_names=['Negative', 'Positive']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, y_pred))
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
