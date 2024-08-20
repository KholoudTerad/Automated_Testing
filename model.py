import re
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class Chatbot:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def validate_input(self, text):
        # Remove any non-alphanumeric characters or symbols
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Limit input length to 100 characters
        text = text[:200]  # text[:200]
        return text

    def tokenize_input(self, text):
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        return inputs

    def prepare_data(self, conversations):
        inputs = [self.tokenize_input(conversation) for conversation in conversations]
        return torch.cat(inputs, dim=1)


    def train_model(self, input_data):
        # Fine-tune the model with input data (not implemented in this example)
        pass

    def generate_response(self, input_text, max_length=50):
        input_text = self.validate_input(input_text)
        input_ids = self.tokenize_input(input_text)
        with torch.no_grad():
            output = self.model.generate(input_ids,
                                         max_length=max_length,
                                         num_return_sequences=1)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response = {"response": response}
        return response
    
if __name__ == "__main__":
    chatbot = Chatbot()

    # Example conversation data
    conversations = [
        "User: Hello! How are you? \n Bot: Im doing well thank you How about you",
        "User: Can you help me with a question? \n Bot: Of course What do you need help with"
    ]

    
    input_data = chatbot.prepare_data(conversations)
    
    # Train model (not implemented in this example)
    chatbot.train_model(input_data)

    # Generate response
    user_input = "User Hi what's your name Bot "
    response = chatbot.generate_response(user_input)
    print("Chatbot: ", response)
