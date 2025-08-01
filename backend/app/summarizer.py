from transformers import pipeline
import re

# Load both BART and T5 summarization pipelines
bart_summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
t5_summarizer = pipeline('summarization', model='t5-base')

def generate_summary(text, model='bart'):
    # Clean and prepare text
    text = text.strip()
    if not text:
        return "No text provided for summarization."
    
    # For very short texts, return as is
    if len(text.split()) < 20:
        return text
    
    try:
        if model == 't5':
            # T5 works better with shorter inputs, so we'll chunk longer texts
            if len(text) > 1000:
                # Split into chunks and summarize each
                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                summaries = []
                for chunk in chunks:
                    input_text = f'summarize: {chunk}'
                    summary = t5_summarizer(input_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                    summaries.append(summary)
                return ' '.join(summaries)
            else:
                input_text = f'summarize: {text}'
                summary = t5_summarizer(input_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        else:
            # BART can handle longer texts better
            if len(text) > 2000:
                # For very long texts, take the first 2000 characters and summarize
                text = text[:2000]
            
            summary = bart_summarizer(text, max_length=200, min_length=80, do_sample=False)[0]['summary_text']
        
        # Clean up the summary
        summary = re.sub(r'\s+', ' ', summary).strip()
        return summary
        
    except Exception as e:
        print(f"Error in summarization: {e}")
        # Fallback: return first 200 characters
        return text[:200] + "..." if len(text) > 200 else text 