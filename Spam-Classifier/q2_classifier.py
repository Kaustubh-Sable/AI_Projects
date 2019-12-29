from sklearn.metrics import confusion_matrix
import argparse

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f1', required=True)
parser.add_argument('-f2', required=True)
parser.add_argument('-o', required=True)

args = vars(parser.parse_args())

train_dataset_filepath = args['f1']
test_dataset_filepath = args['f2']
output_file = args['o']

# collecting all unique words and building the vocabulary
set_of_words = set()

with open(train_dataset_filepath) as fp:
   line = fp.readline()
   count = 1
   while line:
       tokenlist = line.strip().split()
       for i in range(2,len(tokenlist),2):
           set_of_words.add(str(tokenlist[i]))
       line = fp.readline()
       count += 1

count = count-1
count_matrix = [[0 for i in range(len(set_of_words))] for j in range(count)]
labels = [0 for i in range(count)]
list_of_words = list(set_of_words)

# building the count_matrix and list of labels
with open(train_dataset_filepath) as fp:
   line = fp.readline()
   count = 1
   while line:
       tokenlist = line.strip().split()
       labels[count-1] = 1 if tokenlist[1]=='spam' else 0
       for i in range(2,len(tokenlist),2):
           count_matrix[count-1][list_of_words.index(str(tokenlist[i]))] = int(tokenlist[i+1])
       line = fp.readline()
       count += 1

# passing the count_matrix to naive bayes classifier
# older implementation
# model1 = MultinomialNB(alpha=0.5)
# model1.fit(count_matrix,labels)

prior_count = []
prior_count.append(labels.count(0))
prior_count.append(labels.count(1))

freq_count = {}
freq_count[0]={}
freq_count[1]={}

# counting frequency of each word for the class ham and for the class spam separately.
for i in range(len(labels)):
    for j in range(len(count_matrix[0])):
        if j in freq_count[labels[i]]:
            freq_count[labels[i]][j] += count_matrix[i][j]
        elif j not in freq_count[labels[i]]:
            freq_count[labels[i]][j] = count_matrix[i][j]

#####################################################
# making predictions for the test set :

# counting the number of examples in the test set
with open(test_dataset_filepath) as fp:
   line = fp.readline()
   count = 1
   while line:
       line = fp.readline()
       count += 1

count = count-1
test_matrix = [[0 for i in range(len(set_of_words))] for j in range(count)]
test_labels = []

# building count_matrix for test set and list of labels
with open(test_dataset_filepath) as fp:
   line = fp.readline()
   count = 1
   while line:
       tokenlist = line.strip().split()
       if tokenlist[1] == 'spam':
           test_labels.append(1)
       else:
           test_labels.append(0)

       for i in range(2,len(tokenlist),2):
           if str(tokenlist[i]) in list_of_words:
               test_matrix[count-1][list_of_words.index(str(tokenlist[i]))] = int(tokenlist[i+1])
       line = fp.readline()
       count += 1

alpha = 3       # additive for Laplace smoothing
result1 = []
prob0 = 1.0
prob1 = 1.0
# loop over all test instances
for i in range(len(test_matrix)):
    probof0 = 1.0
    probof1 = 1.0
    # for all words in vocabulary
    for j in range(len(test_matrix[0])):
        # calculate P(word|class) for both class if word is in the email using the stored word counts
        if test_matrix[i][j] > 0:
            probof0 *= test_matrix[i][j] * ((float(freq_count[0][j]) + alpha) / (float(prior_count[0]) +
                                                                                  alpha * len(set_of_words)))
            probof1 *= test_matrix[i][j] * ((float(freq_count[1][j]) + alpha) / (float(prior_count[1]) +
                                                                                  alpha * len(set_of_words)))

    # multiply the probability with its class prior probability
    probof0 *= float(prior_count[0])
    probof1 *= float(prior_count[1])

    # argmax to classify the email to ham or spam.
    if probof0 > probof1:
        result1.append(0)
    else:
        result1.append(1)

# making predictions for test set
print "Confusion matrix: \n", confusion_matrix(test_labels,result1)
accuracy = float(100) * float([1 if test_labels[i] == result1[i] else 0 for i in range(1000)].count(1))/float(len(test_labels))
print "Accuracy: ", accuracy

# converting the labels to alphabetical form
result_label = []
for i in range(0,len(result1)):
    if result1[i] == 1:
        result_label.append('spam')
    else:
        result_label.append('ham')

# writing the output predictions to a file.
with open(output_file, "w") as fp:
    for entries in result_label:
        fp.write(entries)
        fp.write("\n")

fp.close()
