import json

def read_data(filename):
    with open(filename) as f:
        examples = json.load(f)
    data = {'train': [], 'dev': [], 'test': []}
    sql_queries = set()
    for example in examples:
        split = example['data']
        tup_pair = (example['question'], example['sql'])
        if split in data:
            data[split].append(tup_pair)
        sql_queries.add(example['sql'])
    return (data, sql_queries)

class CodeModel:
    def __init__(self, labels, training_data):
        self.labels = labels
        self.weights = {}
        for question, _ in training_data:
            for label in labels:
                for feature in self.get_features(question, label):
                    if feature not in self.weights:
                        self.weights[feature] = 0

    def get_features(self, question, label):
        return [(word, label) for word in question.split()]

    def get_score(self, question, label):
        score = 0
        for feature in self.get_features(question, label):
            if feature in self.weights:
                score += self.weights[feature]
        return score

    def update(self, question, label, change):
        for feature in self.get_features(question, label):
            if feature in self.weights:
                self.weights[feature] += change

def find_best_code(question, model):
    best_label = None
    best_score = None
    for label in sorted(model.labels):
        score = model.get_score(question, label)
        if best_score is None or score > best_score:
            best_score = score
            best_label = label
    return best_label

def learn(question, answer, model, find_best_code):
    prediction = find_best_code(question, model)
    if prediction != answer:
        model.update(question, answer, 1)
        model.update(question, prediction, -1)

def get_confusion_matrix(eval_data, model, find_best_code):
    confusion_matrix = {
        (true_label, pred_label): 0
        for true_label in model.labels
        for pred_label in model.labels
    }
    for question, true_answer in eval_data:
        predicted = find_best_code(question, model)
        confusion_matrix[(true_answer, predicted)] += 1
    return confusion_matrix

def calculate_accuracy(confusion_matrix, labels):
    correct = sum(confusion_matrix[(l, l)] for l in labels)
    total = sum(confusion_matrix.values())
    if total == 0:
        return 1.0
    return correct / total

def calculate_precision(confusion_matrix, labels):
    precision_dict = {}
    for label in labels:
        tp = confusion_matrix[(label, label)]
        predicted_as_label = sum(confusion_matrix[(true, label)] for true in labels)
        if predicted_as_label == 0:
            precision_dict[label] = 1.0
        else:
            precision_dict[label] = tp / predicted_as_label
    return precision_dict

def calculate_recall(confusion_matrix, labels):
    recall_dict = {}
    for label in labels:
        tp = confusion_matrix[(label, label)]
        actually_label = sum(confusion_matrix[(label, pred)] for pred in labels)
        if actually_label == 0:
            recall_dict[label] = 1.0
        else:
            recall_dict[label] = tp / actually_label
    return recall_dict

def calculate_macro_f1(confusion_matrix, labels):
    precision = calculate_precision(confusion_matrix, labels)
    recall = calculate_recall(confusion_matrix, labels)
    f1_scores = []
    for label in labels:
        p = precision[label]
        r = recall[label]
        if p + r == 0:
            f1_scores.append(1.0)
        else:
            f1_scores.append(2 * p * r / (p + r))
    return sum(f1_scores) / len(f1_scores)

def main(filename, iterations, read_data, model_maker, learn, find_best_code,
         get_confusion_matrix, calculate_accuracy, calculate_macro_f1):
    data, labels = read_data(filename)
    model = model_maker(labels, data['train'])
    dev_scores = []
    for i in range(iterations):
        for question, answer in data['train']:
            learn(question, answer, model, find_best_code)
        dev_cm = get_confusion_matrix(data['dev'], model, find_best_code)
        dev_scores.append({
            'accuracy': calculate_accuracy(dev_cm, labels),
            'macro-f1': calculate_macro_f1(dev_cm, labels)
        })
        print(f"Iter {i+1} done: accuracy={dev_scores[-1]['accuracy']:.4f}  macro-f1={dev_scores[-1]['macro-f1']:.4f}")
    test_cm = get_confusion_matrix(data['test'], model, find_best_code)
    test_score = {
        'accuracy': calculate_accuracy(test_cm, labels),
        'macro-f1': calculate_macro_f1(test_cm, labels)
    }
    return dev_scores, test_score

if __name__ == '__main__':
    print("Starting...")
    dev_scores, test_score = main(
        'a2-data.json',
        10,
        read_data,
        CodeModel,
        learn,
        find_best_code,
        get_confusion_matrix,
        calculate_accuracy,
        calculate_macro_f1
    )
    print(f"\nTest: accuracy={test_score['accuracy']:.4f}  macro-f1={test_score['macro-f1']:.4f}")
