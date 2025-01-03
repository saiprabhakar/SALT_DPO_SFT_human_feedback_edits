import pandas as pd
from datasets import Dataset

import pandas as pd
import json
import os
def process_dataset(synthetic_data_type):
    data_type_name = synthetic_data_type

    file_path = './data/'+data_type_name+'.json'

    AVL_dataset_base_path = './AVL Dataset/'
    with open(file_path) as json_file:
        data = json.load(json_file)
    if data_type_name in ['gpt4_edits_high_to_low', 'gpt4_edits_low_to_high','turbo_edits_low_to_high']:
        del data['edited_summary']
    else:
        del data['hallucination_instructions']
    train_df = pd.DataFrame.from_dict(data)

    if data_type_name=='gpt4_edits_high_to_low':
        train_df.rename(columns={'reference_summary': 'chosen', 'edited_summary_1': 'rejected'}, inplace=True)
        train_df = train_df.drop('model_summary', axis=1)
    if data_type_name=='gpt4_edits_low_to_high' or data_type_name=='turbo_edits_low_to_high':
        train_df.rename(columns={'model_summary': 'rejected', 'edited_summary_1': 'chosen'}, inplace=True)
        train_df = train_df.drop('reference_summary', axis=1)
        
    def do_stats(x):
        return len(x.split())
    def get_edited_summary(x):
        edit_sum = x.split('Hallucinated Summary: \n\n')[-1]
        edit_sum = edit_sum.split('Hallucinated Summary:\n\n')[-1]
        edit_sum = edit_sum.strip()
        if 'Discharge Instructions:' not in edit_sum:
            edit_sum = 'Discharge Instructions: '+edit_sum
        return edit_sum

    train_df['len'] = train_df['chosen'].apply(do_stats)
    train_df['rejected'] = train_df['rejected'].apply(get_edited_summary)

    train_df = train_df.drop_duplicates(subset=['chosen'], keep='first')
    if not os.path.exists('./data/train'):
        os.makedirs('./data/train')
    if not os.path.exists('./data/eval'):
        os.makedirs('./data/eval')
    if not os.path.exists('./data/test'):
        os.makedirs('./data/test')
    file_path = AVL_dataset_base_path+'valid.json'
    eval_df = pd.read_json(file_path, lines=True)

    file_path = AVL_dataset_base_path+'test.json'
    test_df = pd.read_json(file_path, lines=True)

    train_df.to_csv('./data/train/'+data_type_name+'.csv', index=False)
    eval_df.to_csv('./data/eval/'+data_type_name+'.csv', index=False)
    test_df.to_csv('./data/test/'+data_type_name+'.csv', index=False)

    train_df = pd.read_csv('./data/train/'+data_type_name+'.csv')
    eval_df = pd.read_csv('./data/eval/'+data_type_name+'.csv')
    test_df = pd.read_csv('./data/test/'+data_type_name+'.csv')

    train_df['chosen'] = train_df['article'] + "\n\nGenerate the corresponding Discharge Instructions according to the input article: " + train_df['chosen']
    train_df['rejected'] = train_df['article'] + "\n\nGenerate the corresponding Discharge Instructions according to the input article: " + train_df['rejected']

    eval_df['chosen'] = eval_df['text'] + "\n\nGenerate the corresponding Discharge Instructions according to the input article: " + eval_df['summary']
    eval_df['rejected'] = eval_df['text'] + "\n\nGenerate the corresponding Discharge Instructions according to the input article: " + eval_df['summary']

    test_df['chosen'] = test_df['text'] + "\n\nGenerate the corresponding Discharge Instructions according to the input article: "+ test_df['summary']
    test_df['rejected'] = test_df['text'] + "\n\nGenerate the corresponding Discharge Instructions according to the input article: " + test_df['summary']

    train_df = train_df[['chosen', 'rejected']]
    eval_df = eval_df[['chosen', 'rejected']]
    test_df = test_df[['chosen', 'rejected']]

    sub_eval_df = eval_df.sample(n=128, random_state=42)

    # Convert pandas DataFrames to dictionaries
    train_dict = train_df.to_dict(orient='list')
    eval_dict = eval_df.to_dict(orient='list')
    sub_eval_dict = sub_eval_df.to_dict(orient='list')
    test_dict = test_df.to_dict(orient='list')

    # Create datasets from dictionaries
    train_dataset = Dataset.from_dict(train_dict)
    eval_dataset = Dataset.from_dict(eval_dict)
    sub_eval_dataset = Dataset.from_dict(sub_eval_dict)
    test_dataset = Dataset.from_dict(test_dict)

    if not os.path.exists('./datasets_/'+data_type_name+'/train'):
        os.makedirs('./datasets_/'+data_type_name+'/train')
    if not os.path.exists('./datasets_/'+data_type_name+'/eval'):
        os.makedirs('./datasets_/'+data_type_name+'/eval')
    if not os.path.exists('./datasets_/'+data_type_name+'/sub_eval'):
        os.makedirs('./datasets_/'+data_type_name+'/sub_eval')
    if not os.path.exists('./datasets_/'+data_type_name+'/test'):
        os.makedirs('./datasets_/'+data_type_name+'/test')
    # Replace 'dataset_directory' with the directory where you want to save your datasets
    train_dataset.save_to_disk('./datasets_/'+data_type_name+'/train')
    eval_dataset.save_to_disk('./datasets_/'+data_type_name+'/eval')
    sub_eval_dataset.save_to_disk('./datasets_/'+data_type_name+'/sub_eval')
    test_dataset.save_to_disk('./datasets_/'+data_type_name+'/test')

synthetic_data_types = [
    'gpt4_edits_high_to_low',
    'gpt4_edits_low_to_high',
    'turbo_edits_high_to_low',
    'turbo_edits_low_to_high'
]
for synthetic_data_type in synthetic_data_types:
    process_dataset(synthetic_data_type)
