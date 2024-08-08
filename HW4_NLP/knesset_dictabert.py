from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import sys,os
def replace_masked_token(model, tokenizer, sent):
    try:

        # make encode to the sent and find the index of  [MASK]
        input_ids = tokenizer.encode(sent, return_tensors='pt')
        # find our Mask tkn
        mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
        with torch.no_grad(): #we learned in deep learining that we should put the forward in this

            outputs = model(input_ids)
        # take  the logits
        sentence_logits = outputs.logits[0]
        mask_token_logits = sentence_logits[mask_token_index, :].squeeze()

        # get  the predicted token
        top_token_index = torch.argmax(mask_token_logits, dim=-1)
        top_token = top_token_index.item()

        # get the new token and updated sent
        pred_token = tokenizer.decode([top_token]).strip()
        solved_sent = sent.replace(tokenizer.mask_token, pred_token, 1)
        return solved_sent, pred_token
    except Exception as e:
        print(f'Exception at replace_masked_token: {e}')

if __name__ == '__main__':
    try:
        if len(sys.argv) != 3:
            print('Incorrect input, please enter the folder path and the output path.')
            sys.exit(1)

        mask_file_path = sys.argv[1]
        out_dir = sys.argv[2]

        tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')  #load toknizer
        model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
        model.eval()


        print_prediction = ''
        os.makedirs(out_dir, exist_ok=True)
        out_file_path = os.path.join(out_dir, 'dictabert_results.txt')

        with open(mask_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                pred_tkn_list = []
                curr_sent = line.strip()

                print_prediction +="Original sentence: " +curr_sent
                print_prediction+='\n'
                num_updattes = curr_sent.count('[*]')

                for i in range(num_updattes):
                    curr_sent = curr_sent.replace('[*]','[MASK]',1)
                    curr_sent, predicted_token = replace_masked_token(model,tokenizer,curr_sent)
                    pred_tkn_list.append(predicted_token)

                print_prediction += 'DictaBERT sentence: ' + curr_sent
                print_prediction+='\n'
                print_prediction += 'DictaBERT tokens: ' + ','.join(pred_tkn_list)
                print_prediction+='\n\n'

        with open(out_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(print_prediction)

    except Exception as e:
        print(f'Exception at main: {e}')

