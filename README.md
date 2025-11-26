##  Pipeline Stages

### **1. Preprocessing**
- Removes unnecessary noise  
- URLs, Emails, HTML tags  
- Punctuation & Stopwords  
- Converts text to lowercase  
- Calculates word reduction %  

### **2. Fake/Bias Detection**
- Sensationalism scoring  
- Fake-indicator & Credibility-indicator words  
- Caps-lock intensity  
- Excessive punctuation detection  
- Fake probability score  

### **3. Fact Verification**
- Entity extraction (names, dates, numbers, URLs)  
- Claim identification  
- Knowledge-base matching  
- Source credibility detection  
- Verifiability scoring  

### **4. Headline Rewriting**
- Generates 5 rewritten headline candidates  
- Uses credible, neutral, and cautious templates  
- Entity + action word pairing  
- Ranking system (clarity, credibility, length, non-sensationalism)  

### **5. Final Output Generation**
- Credibility label  
- Bias label  
- Fake-risk label  
- Overall confidence score  
- Warning flags  
- Source references  
- Quality tier (Poor → Excellent)

### PROJECT STRUCTURE:
``` 
 
 fakeheadline-rewriter
│
├──  script.py                     # Main pipeline code
├──  README.md                     # Documentation
├──  requirements.txt              # Dependencies
├──  .gitignore                    # Ignored files
│
├──  sample_data/                  # Sample input dataset
│   └── sample_input.csv
│
└──  outputs/                      # Generated results
    ├── complete_5stage_analysis.csv
    └── complete_5stage_analysis_readable.csv
```
