import math
from collections import Counter
import pandas as pd
import os,sys
class Trigram_LM:

    def __init__(self, corpus_data, protocol_type):
        self.lambda3 = 0.0000001
        self.lambda2 = 0.2999999
        self.lambda1 = 0.7
        self.Unigrams = Counter()
        self.Bigrams = Counter()
        self.Trigrams = Counter()
        self.total_words = 0
        self.dicinoary_size = 0
        self.type = protocol_type
        self.corpus_data = corpus_data
        self.calc_grams()

    def calc_grams(self):
      try:
        for _, row in self.corpus_data.iterrows():
            prot_type = row['protocol_type']
            sentence = row['sentence_text']

            if self.type == prot_type:
                tokens = sentence.strip().split(" ")
                tokens=["<s1>"]+["<s2>"]+tokens
                for idx in range(len(tokens)):
                    self.Unigrams[tokens[idx]] += 1
                    if idx < len(tokens) - 1:
                        self.Bigrams[(tokens[idx], tokens[idx + 1])] += 1
                    if idx < len(tokens) - 2:
                        self.Trigrams[(tokens[idx], tokens[idx + 1], tokens[idx + 2])] += 1


        self.dicinoary_size = len(self.Unigrams)
        self.total_words = sum(self.Unigrams.values())
      except Exception as e:
          print(f"An error occurred while calc grams: {e}")

    def calculate_prob_of_sentence(self, sentence, smoothing="Linear"):
      try:
        tokens = sentence.strip().split()
        total_log_prob = 0.0

        for idx in range(2, len(tokens)):
            curr_trigram = (tokens[idx - 2], tokens[idx - 1], tokens[idx])
            curr_bigram = (tokens[idx - 2], tokens[idx - 1])
            curr_unigram = tokens[idx - 2]

            num_of_curr_trigram = self.Trigrams.get(curr_trigram, 0)
            num_of_curr_bigram = self.Bigrams.get(curr_bigram, 0)
            num_of_curr_unigram = self.Unigrams.get(curr_unigram, 0)
            V = self.dicinoary_size
            trigram_prob = (num_of_curr_trigram + 1) / (num_of_curr_bigram + V)
            bigram_prob = (num_of_curr_bigram + 1) / (num_of_curr_unigram + V)
            unigram_prob = (self.Unigrams[tokens[idx]] + 1) / (self.total_words + V)

            if smoothing == "Linear":
                tmp_prob = self.lambda1 * trigram_prob + self.lambda2 * bigram_prob + self.lambda3 * unigram_prob
            elif smoothing == "Laplace":
                tmp_prob = trigram_prob

            if tmp_prob > 0:
             total_log_prob += math.log(tmp_prob)
            else:
                total_log_prob += float('-inf')

        return total_log_prob
      except Exception as e:
          print(f"An error occurred calculate_prop_of_sentnce: {e}")

    def generate_next_token(self, sentence):
      try:
        max_probality = float('-inf')
        tmp_tkn = None
        sentnce_tokens = sentence.strip().split(' ')
        if("<s1>" not in sentnce_tokens):
            sentnce_tokens=["<s1>"]+["<s2>"]+sentnce_tokens
        if len(sentnce_tokens) > 2:
            sentnce_tokens = sentnce_tokens[-2:]
            sentence = sentnce_tokens[0] + ' ' + sentnce_tokens[1]
        for token in (self.Unigrams.keys()):
          if (("<s1>" not in token) and ("<s2>" not in token)):
            tmp_sentence = sentence + " " + token
            prob = self.calculate_prob_of_sentence(tmp_sentence, "Linear")
            if prob > max_probality:
                max_probality = prob
                tmp_tkn = token
        next_token = tmp_tkn
        return next_token
      except Exception as e:
          print(f"An error occurred while genrate new token: {e}")

    def get_k_n_collocations(self, k, n, corpus_data, type='frequency'):
        try:
            n_grams = Counter()
            for _, row in corpus_data.iterrows():
                    sentence = row['sentence_text']
                    tokens = sentence.strip().split(" ")
                    for i in range(len(tokens) - n + 1):
                        sent_n = tuple(tokens[i:i + n])
                        n_grams[sent_n] += 1
            if type == "frequency":
                k_common_kolk = n_grams.most_common(k)

            elif type == "tfidf":
                num_of_sentences_with_term = Counter()
                num_of_sentences = 0
                total_tfidf = {}
                term_counts = Counter()

                for _, row in corpus_data.iterrows():
                    num_of_sentences += 1
                    sentence = row['sentence_text']
                    tokens = sentence.strip().split(" ")
                    seen_terms = set()

                    for i in range(len(tokens) - n + 1):
                        sent_n = tuple(tokens[i:i + n])
                        seen_terms.add(sent_n)

                    for term in seen_terms:
                        num_of_sentences_with_term[term] += 1

                for _, row in corpus_data.iterrows():
                    sentence = row['sentence_text']
                    tokens = sentence.strip().split(" ")
                    tf_sent = Counter()

                    for i in range(len(tokens) - n + 1):
                        sent_n = tuple(tokens[i:i + n])
                        tf_sent[sent_n] += 1

                    for term, tf in tf_sent.items():
                        tf_last = (tf / (len(tokens) - n + 1))
                        idf = math.log(num_of_sentences / (num_of_sentences_with_term[term] + 1))
                        tfidf = tf_last * idf
                        total_tfidf[term] = total_tfidf.get(term, 0) + tfidf
                        term_counts[term] += 1

                avg_tfidf = Counter()
                for term, tfidf in total_tfidf.items():
                    avg_tfidf[term] = tfidf / term_counts[term]

                k_common_kolk =avg_tfidf.most_common(k)

            tmp_kolk = []
            for tmp, _ in k_common_kolk:
                tmp_kolk.append(tmp)
            k_common_kolk = tmp_kolk

            return k_common_kolk

        except Exception as e:
            print(f"An error occurred while return the k most common kolktsion from len n : {e}")




class Test:
    def __init__(self, corpus_data,path_mask,out_path):
        self.plenary_Trigram = Trigram_LM(corpus_data, "plenary")
        self.committee_Trigram = Trigram_LM(corpus_data, "committee")
        self.path_to_masked_sentnces=path_mask
        self.out_path=out_path

    def complete_sentences(self, type):
      try:
        generated_sentences = []
        genera_tokens = []
        real=[]
        with open(self.path_to_masked_sentnces, 'r', encoding='utf-8') as file:
            for line in file:
                real.append(line)
                last_sent = ""
                new_tokens_for_sent = ""
                parts = line.strip().split('[*]')
                for idx in range(len(parts)):
                    if idx < len(parts) - 1:
                        if type == 'plenary':
                            token = self.plenary_Trigram.generate_next_token(parts[idx])
                            new_tokens_for_sent = new_tokens_for_sent + token + ','
                            last_sent += parts[idx] + token
                            parts[idx] = parts[idx] + token
                        elif type == 'committee':
                            token = self.committee_Trigram.generate_next_token(parts[idx])
                            new_tokens_for_sent = new_tokens_for_sent + token + ','
                            last_sent += parts[idx] + token
                            parts[idx] = parts[idx] + token
                    else:
                        last_sent += parts[idx]
                        new_tokens_for_sent = new_tokens_for_sent[:-1]

                        genera_tokens.append(new_tokens_for_sent)

                generated_sentences.append(last_sent)

        return generated_sentences, genera_tokens,real
      except Exception as e:
          print(f"An error occurred while completing the sentnces : {e}")

    def print_res(self):
      try:

        plenary_complete, plenary_tokens,real = self.complete_sentences("plenary")
        committee_complete, committee_tokens,real = self.complete_sentences("committee")
        os.makedirs(self.out_path, exist_ok=True)
        out_file_path = os.path.join(self.out_path, 'sentences_results.txt')
        with open(out_file_path, 'w', encoding='utf-8') as file:
            for idx in range(len(real)):
                file.write(f"Original sentence: {real[idx].strip()}\n")
                file.write(f"Committee sentence: {committee_complete[idx]}\n")
                file.write(f"Committee tokens: {committee_tokens[idx]}\n")
                prob_committee = self.committee_Trigram.calculate_prob_of_sentence(committee_complete[idx])
                file.write(f"Probability of committee sentence in committee corpus: {prob_committee}\n")
                prob_plenary = self.plenary_Trigram.calculate_prob_of_sentence(committee_complete[idx])
                file.write(f"Probability of committee sentence in plenary corpus: {prob_plenary}\n")
                if prob_plenary >= prob_committee:
                    file.write("This sentence is more likely to appear in corpus: plenary\n")
                else:
                    file.write("This sentence is more likely to appear in corpus: committee\n")

                file.write(f"Plenary sentence: {plenary_complete[idx]}\n")
                file.write(f"Plenary tokens: {plenary_tokens[idx]}\n")
                prob_plenary = self.plenary_Trigram.calculate_prob_of_sentence(plenary_complete[idx])
                file.write(f"Probability of plenary sentence in plenary corpus: {prob_plenary}\n")
                prob_committee = self.committee_Trigram.calculate_prob_of_sentence(plenary_complete[idx])
                file.write(f"Probability of plenary sentence in committee corpus: {prob_committee}\n")
                if prob_plenary >= prob_committee:
                    file.write("This sentence is more likely to appear in corpus: plenary\n")
                else:
                    file.write("This sentence is more likely to appear in corpus: committee\n")
                file.write("\n")
      except Exception as e:
          print(f"An error occurred while print complete sentnces: {e}")


##################################################################################################
##help_func_for _section 2

def print_sec_2(_trigram_committee, _trigram_plenary, corpus_data_committee, corpus_data_plenary,out_dir):
  try:
    os.makedirs(out_dir, exist_ok=True)
    out_file_path = os.path.join(out_dir, 'knesset_collocations.txt')
    with open(out_file_path, 'w', encoding='utf-8') as file:
        file.write("Two-gram collocations:\n")
        file.write("Frequency:\n")

        file.write("Committee corpus:\n")
        committee_collocations_of_length_2 = _trigram_committee.get_k_n_collocations(10, 2, corpus_data_committee, 'frequency')
        for kolk in committee_collocations_of_length_2:
            file.write(f"{kolk}\n")
        file.write("\n")

        file.write("Plenary corpus:\n")
        plenary_collocations_of_length_2 = _trigram_plenary.get_k_n_collocations(10, 2, corpus_data_plenary, 'frequency')
        for kolk in plenary_collocations_of_length_2:
            file.write(f"{kolk}\n")
        file.write("\n")
        #tf
        file.write("TF-IDF:\n")
        file.write("Committee corpus:\n")
        committee_collocations_of_length_2 = _trigram_committee.get_k_n_collocations(10, 2, corpus_data_committee, 'tfidf')
        for kolk in committee_collocations_of_length_2:
            file.write(f"{kolk}\n")
        file.write("\n")
        file.write("Plenary corpus:\n")
        plenary_collocations_of_length_2 = _trigram_plenary.get_k_n_collocations(10, 2, corpus_data_plenary, 'tfidf')
        for kolk in plenary_collocations_of_length_2:
            file.write(f"{kolk}\n")
        file.write("\n")
        ################################################
        file.write("Three-gram collocations:\n")
        file.write("Frequency:\n")
        file.write("Committee corpus:\n")
        committee_collocations_of_length_3 = _trigram_committee.get_k_n_collocations(10, 3, corpus_data_committee, 'frequency')
        for kolk in committee_collocations_of_length_3:
            file.write(f"{kolk}\n")
        file.write("\n")

        file.write("Plenary corpus:\n")
        plenary_collocations_of_length_3 = _trigram_plenary.get_k_n_collocations(10, 3, corpus_data_plenary, 'frequency')
        for kolk in plenary_collocations_of_length_3:
            file.write(f"{kolk}\n")
        file.write("\n")

        file.write("TF-IDF:\n")
        file.write("Committee corpus:\n")
        committee_collocations_of_length_3 = _trigram_committee.get_k_n_collocations(10, 3, corpus_data_committee, 'tfidf')
        for kolk in committee_collocations_of_length_3:
            file.write(f"{kolk}\n")
        file.write("\n")

        file.write("Plenary corpus:\n")
        plenary_collocations_of_length_3 = _trigram_plenary.get_k_n_collocations(10, 3, corpus_data_plenary, 'tfidf')
        for kolk in plenary_collocations_of_length_3:
            file.write(f"{kolk}\n")
        file.write("\n")

        file.write("Four-gram collocations:\n")
        file.write("Frequency:\n")
        file.write("Committee corpus:\n")
        committee_collocations_of_length_4 = _trigram_committee.get_k_n_collocations(10, 4, corpus_data_committee, 'frequency')
        for kolk in committee_collocations_of_length_4:
            file.write(f"{kolk}\n")
        file.write("\n")
        file.write("Plenary corpus:\n")
        plenary_collocations_of_length_4 = _trigram_plenary.get_k_n_collocations(10, 4, corpus_data_plenary, 'frequency')
        for kolk in plenary_collocations_of_length_4:
            file.write(f"{kolk}\n")
        file.write("\n")

        file.write("TF-IDF:\n")
        file.write("Committee corpus:\n")
        committee_collocations_of_length_4 = _trigram_committee.get_k_n_collocations(10, 4, corpus_data_committee, 'tfidf')
        for kolk in committee_collocations_of_length_4:
            file.write(f"{kolk}\n")
        file.write("\n")
        file.write("Plenary corpus:\n")
        plenary_collocations_of_length_4 = _trigram_plenary.get_k_n_collocations(10, 4, corpus_data_plenary, 'tfidf')
        for kolk in plenary_collocations_of_length_4:
            file.write(f"{kolk}\n")
        file.write("\n")
  except Exception as e:
      print(f"An error occurred while find kolkatsiot: {e}")

if __name__ == "__main__":
    try:

        if len(sys.argv) != 4:
            print('Incorrect input, please enter the folder path and the output path.')
            sys.exit(1)

        corpus_file_path = sys.argv[1]
        masked_sentences_path = sys.argv[2]
        output_path=sys.argv[3]

        corpus_data = pd.read_json(corpus_file_path, lines=True, encoding='utf-8')

        corpus_data_plenary = corpus_data[corpus_data['protocol_type'] == 'plenary']
        corpus_data_committee = corpus_data[corpus_data['protocol_type'] == 'committee']
        _trigram_committee = Trigram_LM(corpus_data=corpus_data, protocol_type="committee")
        _trigram_plenary = Trigram_LM(corpus_data=corpus_data, protocol_type="plenary")

        print_sec_2(_trigram_committee, _trigram_plenary, corpus_data_committee, corpus_data_plenary,output_path)
        test = Test(corpus_data,masked_sentences_path,output_path)
        test.print_res()

    except Exception as e:
        print(f"An error occurred in main: {e}")
