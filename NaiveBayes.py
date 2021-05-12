class NaiveBayes():

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X_train: np.array, y_train: np.array):
        self.num_classes = len(np.unique(y_train))
        self.num_instances, self.num_features = X_train.shape
        self.classes = {i:c for i,c in enumerate(np.unique(y_train))}

        # splits X into a list of arrays containing instances of a particular class
        instances_from_class = [X_train[y_train == c] for c in self.classes.values()]

        # initalises log cond probability array
        self.log_cond_by_class = np.zeros((self.num_classes, self.num_features))

        # initalises total_word_count_by_class array
        self.total_word_count_by_class = np.zeros((self.num_classes, 1))

        # initialises num examples by class
        self.num_examples_in_class = np.zeros((self.num_classes, 1))

        for c in range(self.num_classes):
            word_freq_for_class = np.sum(instances_from_class[c], axis=0) + self.alpha
            assert 0 not in word_freq_for_class, 'word_freq_should all be > 0'

            self.total_word_count_by_class[c] = np.sum(word_freq_for_class) 
            assert 0 not in self.total_word_count_by_class[c], 'total_word_count must all be > 0'

            self.log_cond_by_class[c, :] = np.log(word_freq_for_class / self.total_word_count_by_class[c])

            self.num_examples_in_class[c] = instances_from_class[c].shape[0]

        total_word_count = np.sum(self.total_word_count_by_class)

        print(self.total_word_count_by_class)

        self.prior_by_class = np.log(self.num_examples_in_class / self.num_instances)

    
    def predict(self, X_test):
        num_instances = len(X_test)
        y = np.zeros((num_instances, 1), dtype=str)

        for i in range(num_instances):
            p_by_class = np.copy(self.prior_by_class)

            for c in range(self.num_classes):
                for word_i in range(self.num_features):
                    log_cond_prob = self.log_cond_by_class[c][word_i]

                    freq = X_test[i,word_i]
                    p_by_class[c] += log_cond_prob * freq
        
            y[i] = self.classes[np.argmax(p_by_class, axis = 0)[0]]
        return y