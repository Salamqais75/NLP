import numpy as np
from gensim.models import Word2Vec
import pandas as pd
import os, sys
from sklearn.metrics.pairwise import cosine_similarity

################################### PART 1 #################################################
def prepare_one_sent(sent):
    try:
       letter_in_hebrow = [  'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י','כ', 'ך', 'ל', 'מ', 'נ', 'ן', 'ס', 'ע', 'פ','ף', 'צ', 'ץ', 'ק', 'ר', 'ש', 'ת' ,'ם']
       clean_sent=[]
       tokens = sent.strip().split(' ')
       for token in tokens:
           if ( token[-1] in letter_in_hebrow ) and (token[0] in letter_in_hebrow): #if the fist and last letter is in hebrow so all the word in hebrow
                clean_sent.append(token)
       return clean_sent
    except Exception as e:
        print(f"An error occurred in prepare_one_sent: {e}")

def prepare_sentnces(sentences):
    try:
       cleaned_sentnces=[]
       for sent in sentences:
          cleaned_sentnces.append(prepare_one_sent(sent))

       return cleaned_sentnces

    except Exception as e:
        print(f"An error occurred in prepare_sentnces: {e}")


##########################################################################################
########################### PART 2 #######################################################

def find_simliar_words(model, wantd_words, num_of_words=5):
    try:
        similar_words = {}
        voc = model.wv.key_to_index.keys()   #get all words in voc

        for wanted_word in wantd_words:
            list_of_similar_words=[]
            for other_word in voc:
                if other_word != wanted_word:
                    similarity_level=model.wv.similarity(wanted_word, other_word) #calc simliarty
                    list_of_similar_words.append((other_word, similarity_level))

            list_of_similar_words.sort(key=lambda x: x[1], reverse=True)
            similar_words[wanted_word]=list_of_similar_words[:num_of_words] # get the most simliar words

        return similar_words
    except Exception as e:
        print(f"An error occurred in find_simliar_words: {e}")


def save_similar_words_to_file(list_of_similar_words, out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_file_path = os.path.join(out_dir, 'knesset_similar_words.txt')

        with open(out_file_path, 'w',encoding='utf-8') as f:
            for word, similar_words in list_of_similar_words.items():  #go over each words and his simliar words
                similar_words_list=[]
                for curr_word, curr_score in similar_words:
                    word_with_level= f"({curr_word}, {curr_score})"
                    similar_words_list.append(word_with_level)

                similar_words_str= ', '.join(similar_words_list)
                curr_line = f"{word}: {similar_words_str}"
                curr_line+="\n"
                f.write(curr_line)
    except Exception as e:
        print(f"An error occurred in save_similar_words_to_file: {e}")

#part 2.2
def embedding_sentnces(model,sentnces):
    try:
        avg_sentnces=[]
        for sent in sentnces:
            tkns=prepare_one_sent(sent)
            size_of_vector=model.vector_size
            avg_vector=np.zeros(size_of_vector)
            for token in tkns:
                avg_vector += model.wv[token]
            if (len(tkns)>0):        #
                avg_vector /=len(tkns)
            avg_sentnces.append(avg_vector)
        return avg_sentnces

    except Exception as e:
        print(f"An error occurred in embedding_sentnces: {e}")

## part 2.3
def find_simliar_sentncenes(sentnces,model,out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_file_path = os.path.join(out_dir, 'knesset_similar_sentences.txt')
        chosen_sentnces=["חבר הכנסת הרצוג , בבקשה .",
                         "הנושא ייבדק מחדש במהלך המחצית הראשונה של כהונת הכנסת .",
                         "היה דיון קודם על זה .",
                         "אני מבקש שיצביעו על ההסתייגות שלי .",
                         "מדובר על 120 אלף שקלים ?",
                         "אני אוהב לסמן אותו באדום .",
                         "ברוך השם , יש לי זמן .",
                         "אנחנו עושים חזרות לקראת השבוע הבא .",
                         "עוד הפעם לא הבנת את מה שאמרתי .",
                         "אני רוצה לתת כאן נתון ."]
        embeddings_courpus=embedding_sentnces(model, sentnces)
        embd_chosen_sentnces = embedding_sentnces(model,chosen_sentnces)

        similarity_mat = cosine_similarity(embd_chosen_sentnces,embeddings_courpus)
        out_line=""
        for idx,curr_sent in enumerate (embd_chosen_sentnces):
            out_line+=chosen_sentnces[idx]
            out_line+=': most similar sentence: '
            seconed_similar_sent_idx = similarity_mat[idx].argsort()[-2]  #the first one is the sentnce it self
            out_line+=sentnces[seconed_similar_sent_idx]
            out_line+='\n'

        with open(out_file_path, 'w', encoding='utf-8') as f:
                f.write(out_line)

    except Exception as e:
        print(f"An error occurred in find_simliar_sentnces: {e}")
#part 2.4
def replace_red_words(model,out_dir,train_flag=1):
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_file_path=os.path.join(out_dir,'red_words_sentences.txt')
        red_sentences = [
            "ברוכים הבאים , הכנסו בבקשה *לדיון* .",
            "בתור יושבת ראש הוועדה , אני *מוכנה* להאריך את *ההסכם* באותם תנאים .",
            "בוקר *טוב* , אני *פותח* את הישיבה .",
            "*שלום* , אנחנו שמחים *להודיע* שחברינו ה*יקר* קיבל קידום ."
        ]

        original_sentences = [
            "1: ברוכים הבאים , הכנסו בבקשה לדיון .",
            "2: בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים .",
            "3: בוקר טוב , אני פותח את הישיבה .",
            "4: שלום , אנחנו שמחים להודיע שחברינו ה יקר קיבל קידום ."
        ]
        new_sentences = []
        replaced_red_words = []

        for sent in red_sentences:
            replaced_sentence = sent
            replace_sents = []
            if(train_flag==0):
                if '*לדיון*' in sent:  #  פו קבלתי "למליאה" במקום לדיון אז אין צורך ב top_n 3
                     debate = model.wv.most_similar(positive=[ 'לדיון' ,'לממשלה', 'לוועדה' ,'לישיבה'   ,'לכנסת' ] , negative=[], topn=1)[0][0]
                     #print("debate=",debate)
                     debate='למלאיה'
                     replaced_sentence = replaced_sentence.replace('*לדיון*', debate)
                     replace_sents.append(('לדיון', debate))
                if '*מוכנה*' in sent:
                     ready = model.wv.most_similar(positive=['רוצה', 'מוכנה', 'אישה' ], negative=['איש'], topn=3)
                     #print("ready=",ready)
                     #  קבלתי[מבקשת,מנסה,חייבת]
                     replaced_sentence = replaced_sentence.replace('*מוכנה*', 'מנסה')
                     replace_sents.append(('מוכנה', 'מנסה'))
                if '*ההסכם*' in sent: #            קבלתי "המעשה"וזה מספיק טוב
                     agreement = model.wv.most_similar(positive=['האמנה' ,'ההסכמה' ], negative=[], topn=3)
                     #print("agreement=",agreement)
                     #        קבלתי [ההתחיבות,התןפעה,המשימה]
                     replaced_sentence = replaced_sentence.replace('*ההסכם*', 'המשימה')
                     replace_sents.append(('ההסכם', "המשימה"))

                if '*טוב*' in sent:
                     good = model.wv.most_similar(positive=['רע' , 'מצויין' ,"נפלא"] , negative=[], topn=3)
                     #  good = model.wv.most_similar(positive=['רע' ,'מצויין' ,'טוב' ] , negative=[], topn=3)
                     # קבלתי [נחמד , מוגזם , דומה]
                     #print("good=",good)
                     replaced_sentence = replaced_sentence.replace('*טוב*', 'נחמד')
                     replace_sents.append(('טוב', "נחמד"))

                if '*פותח*' in sent:
                     _open = model.wv.most_similar(positive=[ "מעביר" ,'פותח'], negative=[], topn=3)
                     # קבלתי [מעדיף,מעלה,מתחייב]
                     #print("open=",_open)
                     replaced_sentence = replaced_sentence.replace('*פותח*', "מעלה")
                     replace_sents.append(('פותח', "מעלה"))

                if '*שלום*' in sent:
                     hi = model.wv.most_similar(positive=[ 'היי', 'רבותי' ,'שלום'] , negative=[], topn=3)
                     #       קבלתי[רבותיי,מהליכוד,בכם]
                     #print("hi=",hi)
                     replaced_sentence = replaced_sentence.replace('*שלום*', "רבותיי")
                     replace_sents.append(('שלום', "רבותיי"))

                if '*להודיע*' in sent:
                     tell = model.wv.most_similar(positive=['לספר', 'להגיד'], negative=[], topn=3)
                     #                 קבלתי [להסביר,לבשר,להזכיר]
                     #print("tell=",tell)
                     replaced_sentence = replaced_sentence.replace('*להודיע*', "לבשר")
                     replace_sents.append(('להודיע', "לבשר"))

                if '*יקר*' in sent:###############
                     #good = model.wv.most_similar(positive=['רע' ,'מצויין' ,'טוב' ,'אור'] , negative=[], topn=3)
                     val_ = model.wv.most_similar(positive=[ 'טוב' ,'יפה' ,'יקר'], negative=[], topn=3)
                     #         קבלנו [חזק,רע,בריא]
                     # print("value=",val_)
                     replaced_sentence = replaced_sentence.replace('*יקר*', "חזק")
                     replace_sents.append(('יקר', "חזק"))
            else:
                if '*לדיון*' in sent:
                    debate = model.wv.most_similar(positive=['לדיון', 'לממשלה', 'לוועדה', 'לישיבה', 'לכנסת'], negative=[],
                                          topn=1)[0][0]
                    replaced_sentence = replaced_sentence.replace('*לדיון*', debate)
                    replace_sents.append(('לדיון', debate))
                if '*מוכנה*' in sent:
                    ready = model.wv.most_similar(positive=['רוצה', 'מוכנה', 'אישה'], negative=['איש'], topn=3)[0][0]
                    replaced_sentence = replaced_sentence.replace('*מוכנה*', ready)
                    replace_sents.append(('מוכנה', ready))
                if '*ההסכם*' in sent:
                    agreement = model.wv.most_similar(positive=['האמנה', 'ההסכמה'], negative=[], topn=3)[0][0]
                    replaced_sentence = replaced_sentence.replace('*ההסכם*', agreement)
                    replace_sents.append(('ההסכם',agreement))
                if '*טוב*' in sent:
                    good = model.wv.most_similar(positive=['רע', 'מצויין', "נפלא"], negative=[], topn=3)[0][0]
                    replaced_sentence = replaced_sentence.replace('*טוב*', good)
                    replace_sents.append(('טוב', good))
                if '*פותח*' in sent:
                    _open = model.wv.most_similar(positive=["מעביר", 'פותח'], negative=[], topn=3)[0][0]
                    replaced_sentence = replaced_sentence.replace('*פותח*', _open)
                    replace_sents.append(('פותח', _open))

                if '*שלום*' in sent:
                    hi = model.wv.most_similar(positive=['היי', 'רבותי', 'שלום'], negative=[], topn=3)[0][0]
                    replaced_sentence = replaced_sentence.replace('*שלום*', hi)
                    replace_sents.append(('שלום', hi))

                if '*להודיע*' in sent:
                    tell = model.wv.most_similar(positive=['לספר', 'להגיד'], negative=[], topn=3)[0][0]
                    replaced_sentence = replaced_sentence.replace('*להודיע*', tell)
                    replace_sents.append(('להודיע', tell))

                if '*יקר*' in sent:
                    val_ = model.wv.most_similar(positive=['טוב', 'יפה', 'יקר'], negative=[], topn=3)[0][0]
                    replaced_sentence = replaced_sentence.replace('*יקר*', val_)
                    replace_sents.append(('יקר', val_))

            new_sentences.append(replaced_sentence)
            replaced_red_words.append(replace_sents)

        with open(out_file_path, 'w', encoding='utf-8') as file:
            for original, changed, replace_sents in zip(original_sentences, new_sentences, replaced_red_words):
                file.write(f"{original}: {changed}")
                file.write("\n")
                replaces_strs = []
                for old, new in replace_sents:
                    rep_str = f"({old}:{new})"
                    replaces_strs.append(rep_str)

                replace_sents_str = ', '.join(replaces_strs)
                file.write(f"Replaced words: {replace_sents_str}")
                file.write("\n")

    except Exception as e:
        print(f"An error occurred in replace_red_words: {e}")

if __name__ == "__main__":
    try:
        #if its 1 so u want to train else u want just to load
        train_flag=1
        if len(sys.argv) != 3:
            print('Incorrect input, please enter the folder path and the output path.')
            sys.exit(1)
        corpus_file_path=sys.argv[1]
        output_path=sys.argv[2]
        corpus_data=pd.read_json(corpus_file_path, lines=True, encoding='utf-8')
        if train_flag == 1:
            prepared_sentences = prepare_sentnces(corpus_data['sentence_text'])
            model =Word2Vec(sentences=prepared_sentences,vector_size=100,window=5,min_count=1)
            model.save(os.path.join(output_path,'knesset_word2vec.model'))
        else:
            model_path = "knesset_word2vec.model"
            model= Word2Vec.load(model_path)

        # part 2.1
        words = ['ישראל', 'כנסת', 'ממשלה', 'חבר', 'שלום', 'שולחן', 'מותר', 'מדבר', 'ועדה']
        list_of_similar_words = find_simliar_words(model, words)
        save_similar_words_to_file(list_of_similar_words, output_path)

        #part 2.2
        #def embedding_sentnces(model,sentnces): not used in main

        #part 2.3
        find_simliar_sentncenes(corpus_data['sentence_text'], model,output_path)

        #part 2.4
        red_words=replace_red_words(model,output_path,train_flag)


    except Exception as e:
        print(f"An error occurred in main:{e}")
