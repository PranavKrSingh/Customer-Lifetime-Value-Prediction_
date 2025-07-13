
from transformers import pipeline

_generator = pipeline('text-generation', model='distilgpt2')

def gen_answer(context:str, question:str):
    prompt = f"""You are a retail analytics assistant. Use the context to answer the question.
    Context:
    {context}

    Question: {question}
    Answer:"""
    resp = _generator(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']
    return resp.split('Answer:')[-1].strip()
