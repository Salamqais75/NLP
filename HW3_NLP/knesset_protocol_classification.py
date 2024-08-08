import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd 
import numpy as np 
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
import os, sys


CHUNK_SIZE = 5 # Number of sentences in each chunk
K = 3 # Number of neighbors for KNN
TOP_N_WORDS = 3000  # Number of most used words to consider for custom features    


try:
    def calculate_word_counts(text):
        all_words = ' '.join(text).split()
        word_counts = Counter(all_words)
        return word_counts
    
except Exception as e:
    print(f'Error in calculate_word_counts: {e}')


try:
    def calculate_top_words(text, top_n=TOP_N_WORDS):
        all_words = ' '.join(text).split()
        word_counts = Counter(all_words)
        top_words = [word for word, _ in word_counts.most_common(top_n)]
        return top_words

except Exception as e:
    print(f'Error in calculate_top_words: {e}')


try:
    # Function to find the words with a count difference greater than 20
    def find_significant_word_differences(committee_counts, plenary_counts, threshold=20):
        significant_words = {}
        for word, count in committee_counts.items():
            if word in plenary_counts:
                diff = abs(count - plenary_counts[word])
                if diff > threshold:
                    significant_words[word] = {
                        'committee': count,
                        'plenary': plenary_counts[word],
                        'difference': diff
                    }
        return significant_words

except Exception as e:
    print(f'Error in find_significant_word_differences: {e}')


try:
    def get_Chunks(df):
        chunk_size = CHUNK_SIZE
        combined_rows = []

        # Process committee protocol type
        committee_group = df[df['protocol_type'] == 'committee']
        committee_sentences = committee_group['sentence_text'].tolist()
        num_committee_chunks = len(committee_sentences) // chunk_size
        for i in range(num_committee_chunks):
            chunk_sentences = ' '.join(committee_sentences[i*chunk_size:(i+1)*chunk_size])
            kneeset_number = committee_group.iloc[i*chunk_size]['knesset_number']
            combined_rows.append({'protocol_type': 'committee', 'sentence_text': chunk_sentences, 'knesset_number': kneeset_number})

        # Process plenary protocol type
        plenary_group = df[df['protocol_type'] == 'plenary']
        plenary_sentences = plenary_group['sentence_text'].tolist()
        num_plenary_chunks = len(plenary_sentences) // chunk_size
        for i in range(num_plenary_chunks):
            chunk_sentences = ' '.join(plenary_sentences[i*chunk_size:(i+1)*chunk_size])
            kneeset_number = plenary_group.iloc[i*chunk_size]['knesset_number']
            combined_rows.append({'protocol_type': 'plenary', 'sentence_text': chunk_sentences, 'knesset_number': kneeset_number})
        return pd.DataFrame(combined_rows)

except Exception as e:
    print(f'Error in get_Chunks: {e}')


try:
    def extract_custom_features(chunks, custom_words, kneeset_numbers):
        features = []
        # Calculate number of occurrences of custom words in each sentence
        
        for i, chunk in enumerate(chunks):
            word_counts = Counter(chunk.split())
            chunk_features = [word_counts.get(word, 0) for word in custom_words]
            chunk_features.append(kneeset_numbers[i])
            #chunk_features.append(len(chunk)) # Length of chunk
            features.append(chunk_features)

        return np.array(features)

except Exception as e:
    print(f'Error in extract_custom_features: {e}')


if __name__ == '__main__':
    try:

        if len(sys.argv) != 4:
            print("Error! please enter the following: python knesset_protocol_classification.py <corpus_file> <sentences_file> <output_dir>")
            sys.exit(1)

        corpus_file = sys.argv[1]
        sentences_file = sys.argv[2]
        output_dir = sys.argv[3]

        np.random.seed(42)
        random.seed(42)


        # Part 1: Load the data
        df = pd.read_json(corpus_file, lines=True, encoding='utf-8')

        # Part 2: Split into chunks
        df = get_Chunks(df)


        # Part 3: down sampling
        # Get the indexes of the down sample
        committee_indexes = df[df['protocol_type'] == 'committee'].index
        plenary_indexes = df[df['protocol_type'] == 'plenary'].index
        #print("Len of original committee", len(committee_indexes))
        #print("Len of original plenary",len(plenary_indexes))

        # Down sample the majority class
        down_sampled_indexes = np.random.choice(plenary_indexes, size=len(committee_indexes), replace=False)
        #print("Len of new plenary",len(down_sampled_indexes))

        # Combine the indexes
        combined_indexes = np.concatenate([committee_indexes, down_sampled_indexes])
        # Get the down sampled dataframe
        df = df.loc[combined_indexes].reset_index(drop=True)


        '''
        ###################################################################3

        # Separate sentences by protocol type
        committee_sentences = df[df['protocol_type'] == 'committee']['sentence_text'].tolist()
        plenary_sentences = df[df['protocol_type'] == 'plenary']['sentence_text'].tolist()

        # Calculate word count for each protocol type
        committee_word_counts = calculate_word_counts(committee_sentences)
        plenary_word_counts = calculate_word_counts(plenary_sentences)

        # Find words with a count difference greater than threshold
        significant_words = find_significant_word_differences(committee_word_counts, plenary_word_counts, threshold=0)
        # sort the significant words by the difference
        significant_words = dict(sorted(significant_words.items(), key=lambda item: item[1]['difference'], reverse=True))

        # Print and save the significant words
        with open('significant_words3.txt', 'w', encoding='utf-8') as file:
            for word, data in significant_words.items():
                line = (f'{word}: Committee={data["committee"]}, '
                        f'Plenary={data["plenary"]}, Difference={data["difference"]}')
                file.write(line + '\n')

        ###################################################################3
        '''


        # Part 4.1: Feature vector

        chunks = df['sentence_text']
        kneeset_numbers = df['knesset_number']
        count_vectorizer = CountVectorizer()
        count_vectorizer.fit(chunks)
        count_features = count_vectorizer.transform(chunks)


        tfid_vectorizer = TfidfVectorizer()
        tfid_vectorizer.fit(chunks)
        tfid_features = tfid_vectorizer.transform(chunks)

        # Part 4.2: Custom features
        custom_words = ['הכנסת', 'חבר', 'אדוני','חברי', 'אנחנו','היושב-ראש', 'אני','הצעת', 'חוק','ישראל','הממשלה','השר','צריך','בבקשה']
        #custom_features = extract_custom_features(chunks, top_words)
        custom_features = extract_custom_features(chunks, custom_words, kneeset_numbers)

    


        for current_features in [custom_features, count_features, tfid_features]:
            # Part 5: Training models
            labels = df['protocol_type']
            if current_features is count_features:
                curr = 'Count features'
            elif current_features is tfid_features:
                curr = 'TFID features'
            else:
                curr = 'Custom features'
                
            #print('Train validation with '+curr)


            # Models
            KNN = KNeighborsClassifier(K)
            LR = LogisticRegression(max_iter=10000)  # Added max_iter to ensure convergence

            # 5 fold cross validation

            knn_scores = cross_val_score(KNN, current_features, labels, cv=5)
            lr_scores = cross_val_score(LR, current_features, labels, cv=5)

            #print(f'KNN cross-validation scores: {knn_scores.mean()}')
            #print(f'LR cross-validation scores: {lr_scores.mean()}')

            y_pred = cross_val_predict(KNN, current_features, labels, cv=5)
            #print(f'KNN with cross validation:')
            #print(classification_report(labels, y_pred))

            y_pred = cross_val_predict(LR, current_features, labels, cv=5)
            #print(f'LR with cross validation:')
            #print(classification_report(labels, y_pred))


            # Train Test Split
            X_train, X_test, y_train, y_test = train_test_split(current_features, labels, test_size=0.1, random_state=42,stratify=labels)
            KNN.fit(X_train,y_train)
            LR.fit(X_train, y_train)
            if current_features is tfid_features:
                best_model = LR

            #print(f'KNN with split:')
            y_pred = KNN.predict(X_test)
            #print(classification_report(y_test, y_pred))
            
            #print(f'LR with split:')
            y_pred = LR.predict(X_test)
            #print(classification_report(y_test, y_pred))
            #print('--------------------------------------------------')


        '''
        for k in [3,5,10,15,20,25,50,100]:
            print(f'KNN with {k} neighbors and count features:')
            KNN = KNeighborsClassifier(k)
            knn_scores = cross_val_score(KNN, count_features, labels, cv=5)
            print(f'KNN with {k} neighbors cross-validation scores: {knn_scores.mean()}')

            X_train, X_test, y_train, y_test = train_test_split(count_features, labels, test_size=0.1, random_state=42,stratify=labels)
            KNN.fit(X_train,y_train)
            print(f'KNN with {k} neighbors with split:')
            y_pred = KNN.predict(X_test)
            print(classification_report(y_test, y_pred))

            print('--------------------------------------------------')
            
            print(f'KNN with {k} neighbors and tfidf features:')
            KNN = KNeighborsClassifier(k)
            knn_scores = cross_val_score(KNN, tfid_features, labels, cv=5)
            print(f'KNN with {k} neighbors cross-validation scores: {knn_scores.mean()}')

            X_train, X_test, y_train, y_test = train_test_split(tfid_features, labels, test_size=0.1, random_state=42,stratify=labels)
            KNN.fit(X_train,y_train)
            print(f'KNN with {k} neighbors with split:')
            y_pred = KNN.predict(X_test)
            print(classification_report(y_test, y_pred))

            print('--------------------------------------------------')
        '''


        # Part 6: Classification
        # Choose the best model and feature vector

        with open(sentences_file, 'r', encoding='utf-8') as file:
            sentences = file.readlines()
            new_count_features = tfid_vectorizer.transform(sentences)
            predictions = best_model.predict(new_count_features)
            text = ''

        for i, prediction in enumerate(predictions):
            text += prediction + '\n'
        with open(os.path.join(output_dir, 'classification_results.txt'), 'w') as write_file:
            write_file.write(text)


    except Exception as e:
        print(f'Error in main: {e}')    