import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from gensim.models import Word2Vec

def prepare_one_sent(sent):
    try:
       letter_in_hebrow = [  'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י','כ', 'ך', 'ל', 'מ', 'ם', 'נ', 'ן', 'ס', 'ע', 'פ','ף', 'צ', 'ץ', 'ק', 'ר', 'ש', 'ת']
       new_sent=[]
       tokens = sent.strip().split(' ')
       for token in tokens:
           if ( token[-1] in letter_in_hebrow ) and (token[0] in letter_in_hebrow):
                new_sent.append(token)
       return new_sent
    except Exception as e:
        print(f"An error occurred in prepare_one_sent: {e}")

def embeddings_sentence(model,sentnces):
    try:
        avg_sentnces=[]
        for sent in sentnces:
            tkns=prepare_one_sent(sent)
            size_of_vector=model.vector_size
            avg_vector=np.zeros(size_of_vector)
            for token in tkns:
                avg_vector += model.wv[token]
            if (len(tkns)>0):
                avg_vector /=len(tkns)
            avg_sentnces.append(avg_vector)
        return avg_sentnces

    except Exception as e:
        print(f"An error occurred in embeddings_sentence: {e}")




def get_Chunks(df, chunk_size=1):
    combined_rows = []

    committee_group = df[df['protocol_type'] == 'committee']
    committee_sentences = committee_group['sentence_text'].tolist()
    num_committee_chunks = len(committee_sentences) // chunk_size
    for i in range(num_committee_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_sentences = ' '.join(committee_sentences[start_idx:end_idx])
        knesset_number = committee_group.iloc[start_idx]['knesset_number']
        combined_rows.append({'protocol_type': 'committee', 'sentence_text': chunk_sentences, 'knesset_number': knesset_number})

    plenary_group =df[df['protocol_type'] == 'plenary']
    plenary_sentences= plenary_group['sentence_text'].tolist()
    num_plenary_chunks= len(plenary_sentences) // chunk_size
    for i in range(num_plenary_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk_sentences = ' '.join(plenary_sentences[start_idx:end_idx])
        knesset_number = plenary_group.iloc[start_idx]['knesset_number']
        combined_rows.append({'protocol_type': 'plenary', 'sentence_text': chunk_sentences, 'knesset_number': knesset_number})

    return pd.DataFrame(combined_rows)


def class_method(model,corpus_data, chunks_sizes=[1, 3, 5], num_of_neighbors=5):
    for curr_chunk_size in chunks_sizes:
        print(f"chunk size:{curr_chunk_size}")
        print(f"num_of_neighbors:{num_of_neighbors}")

        # build chunks
        corpus_chunks =get_Chunks(corpus_data, curr_chunk_size)

        #balance the ckunks
        committee_indexes= corpus_chunks[corpus_chunks['protocol_type'] == 'committee'].index
        plenary_indexes= corpus_chunks[corpus_chunks['protocol_type'] == 'plenary'].index
        down_sampled_indexes= np.random.choice(plenary_indexes, size=len(committee_indexes), replace=False)
        combined_indexes = np.concatenate([committee_indexes, down_sampled_indexes])
        corpus_chunks =corpus_chunks.loc[combined_indexes].reset_index(drop=True)

        labels= corpus_chunks['protocol_type']

        embeddings= embeddings_sentence(model, corpus_chunks['sentence_text'])


        # split  data to train and test
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.1, random_state=42, stratify=labels)

        # train knn
        knn =KNeighborsClassifier(n_neighbors=num_of_neighbors)
        knn.fit(X_train, y_train)

        # predict and evaluate
        y_pred=knn.predict(X_test)

        print(classification_report(y_test, y_pred))



if __name__ == "__main__":
    try:
        if len(sys.argv) != 3:
            print('Incorrect input, please enter the folder path and the output path.')
            sys.exit(1)

        corpus_file_path = sys.argv[1]
        model_path = sys.argv[2]

        corpus_data = pd.read_json(corpus_file_path, lines=True, encoding='utf-8')
        model = Word2Vec.load(model_path)

        class_method(model, corpus_data)

    except Exception as e:
        print(f"An error occurred in main: {e}")