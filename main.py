import os
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoImageProcessor, AutoModelForImageClassification
# from flask import Flask, send_file
from flask import Flask, jsonify, request
from pyngrok import ngrok
from PIL import Image

text_generator_model = "./controlled-food-recipe-generation"
# text_generator_tokenizer = "./controlled-food-recipe-generation"


model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained(text_generator_model)

app = Flask(__name__)

items = [
    {'id':  1, 'name': 'Item One'},
    {'id':  2, 'name': 'Item Two'}
]

@app.route('/items', methods=['GET'])
def get_items():
    return jsonify(items)

# @app.route("/")
# def index():
#     return send_file('web/index.html')


# Functions
def convert_tokens_to_string(tokens):
    """
    Converts a sequence of tokens (string) into a single string.
    """
    if tokens is None:
        return ""

    cleaned_tokens = [token for token in tokens if token is not None]
    text = tokenizer.decode(cleaned_tokens, skip_special_tokens=True) if cleaned_tokens else ""

    return text

def generate_text(prompt):
    # Load the fine-tuned model
  
    # Get the custom EOS token ID from the tokenizer
    custom_eos_token_id = tokenizer.encode('<RECIPE_END>', add_special_tokens=False)[0]
    # Set the custom EOS token ID in the model configuration
    model.config.eos_token_id = custom_eos_token_id

    model.eval()

    # Generate text
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=500, num_return_sequences=1)

    # Convert tokens to string
    generated_text = convert_tokens_to_string(output[0])

    # Replace '<' with '\n<' to create a new line at each tag
    generated_text = generated_text.replace('<', '\n<')

    return generated_text



@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    data = request.json
    prompt = data['prompt']
    print(prompt)

    # Generate text using the fine-tuned model
    generated_text = generate_text(prompt)

    return jsonify({'generated_text': generated_text})





#image classificaiton route form vit image classificaiton model
   
food_classification_model = "google/vit-base-patch16-224-in21k"
# food_classification_model = "./food-image-classification"


# Load image processor and model
image_classify_model = AutoModelForImageClassification.from_pretrained(food_classification_model)
image_processor = AutoImageProcessor.from_pretrained(food_classification_model)

from transformers import pipeline

@app.route('/classify', methods=['POST'])
def classify_image():
    if request.method == 'POST':
        # Check if the request contains an image file
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'})
        
         # Read the image file
        image_file = request.files['image']

        try:
            image_data=image_file.read()
            # print(image_file,"this is email file")
            image_pil=Image.open(image_file)
        except Exception as e:
          print(e,"this is error")
          return 'Failed to read or convert the image to PIL FORMAT', 400

        # Process the image
        inputs = image_processor(image_pil, return_tensors="pt")

        # Classify the image
        with torch.no_grad():
            logits = image_classify_model(**inputs).logits
        predicted_label_id = logits.argmax(-1).item()
        predicted_label = image_classify_model.config.id2label[predicted_label_id]

        # Return the predicted label
        return jsonify({'predicted_label': predicted_label})


if __name__ == "__main__":
    app.run(debug=True,port=int(os.environ.get('PORT', 8000)))

