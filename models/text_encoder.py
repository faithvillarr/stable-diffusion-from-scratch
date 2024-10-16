'''
    Rather than going through the costly process of training a full word-embedding model,
    this stable diffusion model will use OpenAI's CLIP model to tokenize prompts and 
    embed them. 
'''

# .env/Scripts/activate

from transformers import CLIPTokenizer, CLIPTextModel
import sys
import torch

class TextEncoder:
    def __init__(self, model_name='openai/clip-vit-base-patch32'):
        # Create CLIP instances
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

    def encode(self, text_prompt):
        # Get text embeddings of prompt from CLIP
        inputs = self.tokenizer(text_prompt, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embeddings = self.model(**inputs).last_hidden_state
        return text_embeddings

if __name__ == "__main__":
    # To test.
    encoder = TextEncoder()
    prompt = "A cat."
    if len(sys.argv) > 1:
        prompt = sys.argv[1]

    text_embeddings = encoder.encode(prompt)
    print(text_embeddings.shape)