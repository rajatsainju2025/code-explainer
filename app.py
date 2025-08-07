import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Load the fine-tuned model and tokenizer
model_dir = "./results"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

def explain_code(code_snippet):
    """
    Takes a Python code snippet and returns model's explanation.
    """

    # 2. Format the input using the same prompt structure as in training
    prompt = f"Explain this Python code: ```python\n{code_snippet}\n```\nExplanation:"

    # 3. Tokenize the input and generate a response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id # Set pad_token_id to eos_token_id
    )

    # 4. Decode the generated response to a string
    # We skip special tokens to avoid printing them, and we slice to remove the input prompt from the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    explaination = generated_text[len(prompt):].strip()

    return explaination

# 5. Create the Gradio Interface
iface = gr.Interface(
    fn=explain_code,
    inputs=gr.Code(language="python", label="Python Code"),
    outputs=gr.Textbox(label='Explanation'),
    title="🐍 Python Code Explainer",
    description="Enter a snippet of Python code and see an AI-generated explanation. This app uses a fine-tuned 'distilgpt2' model.",
    flagging_mode="never"
)

# Launch the app
if __name__ == "__main__":
    iface.launch()