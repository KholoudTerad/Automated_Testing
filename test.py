import pytest
import torch
from model import Chatbot

@pytest.fixture
def chatbot():
    return Chatbot()

# Test input with symbols and exceeding 100 characters
def test_validate_input():
    chatbot = Chatbot()
    test_input = "User: Hello! How are you? I'm doing well, thank you! Can you help me with a question? This input exceeds 100 characters, so it should be truncated. !@#$%^&*()_+=-{}[]|:;'<>?,./"
    expected_output = "User Hello How are you Im doing well thank you Can you help me with a question This input exceeds 100 characters so it should be truncated "
    assert chatbot.validate_input(test_input) == expected_output, "Input validation failed"


# Tokenize input
def test_tokenize_input():
    chatbot = Chatbot()
    input_text = "User Hello How are you Im doing well thank you Can you help me with a question This input exceeds 100 characters so it should be truncated "
    tokenized_input = chatbot.tokenize_input(input_text)
    assert tokenized_input.shape == torch.Size([1, 29]), "Tokenization failed"


# Test generating response for a long input
def test_long_input(chatbot):
    input_text = "User: Could you provide some information about your company's products and services?"
    response = chatbot.generate_response(input_text)
    assert isinstance(response, dict), "Response should be a dictionary"
    assert 'response' in response, "Response should contain 'response' key"
    assert len(response['response']) > 0, "Generated response should not be empty"
  


# Test generating multiple responses
def test_multiple_responses_identical(chatbot):
    input_text = "User: I want to know more about the university\nBot:"
    responses = [chatbot.generate_response(input_text) for _ in range(5)]
    assert len(set(response['response'] for response in responses)) == 1, "Generated responses should be the same"

if __name__=="__main__":
    pytest.main(["-x", "test.py"])