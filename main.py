import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load model
model_path = "ayo-folashade1/contract"


def run_prediction(question_texts, context_text, model_path):
    # Load tokenizer and model based on model_path
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    answers = []

    for question_text in question_texts:
        # Tokenize context and question
        inputs = tokenizer.encode_plus(question_text, context_text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        # Process logits to obtain answer
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())
        answer_start = torch.argmax(start_logits, dim=1).squeeze().tolist()
        answer_end = torch.argmax(end_logits, dim=1).squeeze().tolist()

        # Ensure answer_start and answer_end are lists
        answer_start = [answer_start] if isinstance(answer_start, int) else answer_start
        answer_end = [answer_end] if isinstance(answer_end, int) else answer_end

        # Iterate over each input and extract the corresponding answer
        for start, end in zip(answer_start, answer_end):
            answer = tokenizer.decode(input_ids[0, start:end+1], skip_special_tokens=True)
            answers.append(answer)

    return answers


def main():
    st.title("Contract Question-Answering")

    context = st.text_area("Enter Contract Text", height=200)
    questions = st.text_area("Enter Questions (one per line)", height=100)

    if st.button("Generate Predictions"):
        questions = [q.strip() for q in questions.split("\n") if q.strip()]  # Remove leading/trailing whitespace and exclude empty lines

        if context and questions:  # Check if context and questions are provided
            answers = run_prediction(questions, context, model_path=model_path)  # Pass questions as a list
            for question, answer in zip(questions, answers):
                st.write("Question:", question)
                st.write("Answer:", answer)
        else:
            st.write("Please enter a valid contract text and questions.")


if __name__ == "__main__":
    main()
