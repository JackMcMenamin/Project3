import os
import re
import json
from datasets import load_dataset

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?]) +')

def main():
    # Load 180 Pereira concepts
    with open('data/pereira_2018/Pereira_Materials/stimuli_180concepts.txt', 'r') as f:
        pereira_words = {line.strip().lower() for line in f if line.strip()}
    print(f"Loaded {len(pereira_words)} Pereira concepts.")

    collected_sentences = {w: [] for w in pereira_words}
    
    # 1. Pull existing sentences from wiki_targetwords.jsonl
    if os.path.exists('data/wiki_targetwords.jsonl'):
        with open('data/wiki_targetwords.jsonl', 'r') as f:
            for line in f:
                data = json.loads(line)
                word = data['word']
                if word in pereira_words:
                    collected_sentences[word] = data['sentences']
                    
    missing_words = {w for w, sents in collected_sentences.items() if len(sents) < 60}
    print(f"Found existing sentences for {len(pereira_words) - len(missing_words)} words.")
    print(f"Need to stream Wikipedia for {len(missing_words)} missing words.")
    
    if missing_words:
        print("Streaming Wikipedia...")
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        
        articles_processed = 0
        words_completed = 0
        target_sentences = 60
        
        for article in dataset:
            text = article['text']
            raw_lines = text.split('\n')
            sentences = []
            for line in raw_lines:
                line = line.strip()
                if not line: continue
                sents = _SENT_SPLIT_RE.split(line)
                for s in sents:
                    s = s.strip()
                    if s and s.endswith(('.', '!', '?')):
                        if len(s.split()) >= 5:
                            sentences.append(s)
            
            words_found_in_article = set()
            for s in sentences:
                s = s.strip()
                if not s: continue
                if "Notes  References  Sources" in s:
                    continue
                if len(s.split()) > 100:
                    continue
                
                words_in_s = set(re.findall(r'\b[a-z]+(?:-[a-z]+)*\b', s.lower()))
                found = words_in_s.intersection(missing_words)
                
                for w in found:
                    if w not in words_found_in_article and len(collected_sentences[w]) < target_sentences:
                        collected_sentences[w].append(s)
                        words_found_in_article.add(w)
                        if len(collected_sentences[w]) == target_sentences:
                            missing_words.remove(w)
                            words_completed += 1
                            
            articles_processed += 1
            if articles_processed % 1000 == 0:
                print(f"Processed {articles_processed} articles. Missing words completed: {words_completed}")
                
            if not missing_words or missing_words == {"argumentatively"}:
                break
                
    output_path = os.path.join("data", "wiki_pereira.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for w, sents in collected_sentences.items():
            if sents:
                json_line = json.dumps({"word": w, "sentences": sents})
                f.write(json_line + "\n")
                
    print(f"Finished extracting sentences across {len(pereira_words)} target words.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
