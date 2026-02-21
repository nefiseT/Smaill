# w this file, i cleaned unnecessary data from data texts in data\raw folder

import re   #reguler expressions
import os

def simple_clean(text):
    text = text.encode("ascii", "ignore"). decode() #remove non ascii characters
    text = re.sub(r'\s+', ' ', text)        #to reduce space
    return text.strip()

def process_all_files(data_dir):
    combined_chaos = ""
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                raw_content = f.read()
                cleaned = simple_clean(raw_content)
                combined_chaos += cleaned + " "

    with open ("data/training_input.txt" , "w") as f:
        f.write(combined_chaos)
        print(f"Chaos made character total: {len(combined_chaos)}")

    chars = sorted(list(set(combined_chaos)))
    vocab_size = len(chars)
    print("characters: ")
    print(''.join(chars))
    print("vocab size:" ,vocab_size)

if __name__ == "__main__":
    process_all_files("data/raw/")
    
