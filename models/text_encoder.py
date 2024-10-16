'''
    Rather than going through the costly process of training a full word-embedding model,
    this stable diffusion model will use OpenAI's CLIP model to tokenize prompts and 
    embed them. 
'''

# .env/Scripts/activate

from transformers import CLIPTokenizer, CLIPTextModel

class TextEncoder:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

    def encode(self, text_prompt):
        inputs = self.tokenizer(text_prompt, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.model(**inputs).last_hidden_state
        return text_embeddings
