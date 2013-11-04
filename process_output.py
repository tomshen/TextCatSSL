import csv

pred_labels = {}
with open('data/20_newsgroups/label_prop_output', 'rb') as f:
    datareader = csv.reader(f, delimiter='\t')
    for row in datareader:
        try:
            doc = int(row[0])
            label_info = row[3].split(' ')
            try:
                label = int(label_info[0])
                weight = float(label_info[1])
            except:
                label = int(label_info[2])
                weight = float(label_info[3])

            pred_labels[doc] = (label, weight)
        except:
            # a word, not a doc (contains hash), or if seed label
            continue

# print pred_labels

true_labels = {}
num_pred = 0
num_correct = 0
with open('data/20_newsgroups/test.label', 'r') as f:
    curr_doc = 1
    for label in f:
        if curr_doc in pred_labels:
            num_pred += 1
            if int(label) == pred_labels[curr_doc][0]:
                num_correct += 1
        curr_doc += 1

print '%d correct out of %d' % (num_correct, num_pred)