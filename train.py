import torch
from datasets import Dataset
from transformer import AutoTokenizer, AutoModelForCausalLLM, TrainingArguments, Trainer

def train_model():

    # 1. Define the dataset
    # In a real-world scenario, you would load the from a file (e.g., a JSON or CSV)
    # For this example, we define it directly in the code

    data = [
        {"code": "def add(a, b):\n    return a + b", "explanation": "This is a Python function named 'add' that takes two arguments, 'a' and 'b', and returns their sum."},
        {"code": "x = [1, 2, 3]\nprint(x[0])", "explanation": "This code initializes a list named 'x' with three numbers. It then prints the first element of the list, which is 1."},
        {"code": "for i in range(3):\n    print(i)", "explanation": "This is a 'for' loop that iterates three times. It will print the numbers 0, 1, and 2, each on a new line."},
        {"code": "import math\nprint(math.sqrt(16))", "explanation": "This code imports Python's built-in 'math' module. It then calculates and prints the square root of 16, which is 4.0."},
        {"code": "def greet(name):\n    return f'Hello, {name}!'", "explanation": "This defines a function 'greet' that takes one argument, 'name'. It returns a formatted string that says hello to the provided name."},
        {"code": "a = 5\nb = 10\na, b = b, a", "explanation": "This code swaps the values of two variables. Initially, 'a' is 5 and 'b' is 10. After execution, 'a' becomes 10 and 'b' becomes 5."},
        {"code": "d = {'key': 'value'}\nprint(d['key'])", "explanation": "This initializes a dictionary 'd' with one key-value pair. It then prints the value associated with the key 'key', which is 'value'."},
    ]

    # Convert the list of dicts to a Hugging Face Dataset object
    dataset = Dataset.from_list(data)

    # 2. Initialize Tokenizer and Model
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLLM.from_pretrained(model_name)

    # GPT-2 doesn't have a default pad token, so we set it to the end-of-sentence token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Preprocess the Data
    def tokenize_function(examples):
        # We create a structured prompt for the model
        prompt = f"Explain this Python Code: ```python\n{examples['code']}\n```\nExplanation:"
        #The model's target is the explanation itself
        text = prompt + examples['explanation']
        return tokenizer(text, truncation=True, padding="max_length", max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=False)

    # We need to explicitly set the 'labels' for the language modeling task
    def set_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(set_labels, batched=False)

    # 4. Set Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",             # Directory to save the model
        num_train_epochs=50,                # More epochs for a small dataset
        per_device_train_batch_size=2,      # Batch size
        warmup_steps=10,                    # Warmup steps
        weight_decay=0.01,                  # Weight decay
        logging_dir="./logs",               # Directory for logs
        logging_steps=10
    )

    # 5. Initialize Trainer and Train
    trainer = Trainer(
        model-model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("Start model training...")
    trainer.train()
    print("Training finished!")

    # 6. Save the final model
    trainer.save_model("./results")
    print("Model saved to ./results")

