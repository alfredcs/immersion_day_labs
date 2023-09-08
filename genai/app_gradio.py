import copy
import glob
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import gradio as gr
import PIL
import torch

from gradio import processing_utils
from gradio_client.client import DEFAULT_TEMP_DIR
from text_generation import Client
from transformers import AutoProcessor


MODELS = [
    "HuggingFaceM4/idefics-9b-instruct",
    # "/home/alfred/models/idefics-9b-instruct",
    "HuggingFaceM4/idefics-80b-instruct",
    "local/idefics-9b-instruct",
]

API_PATHS = {
    "HuggingFaceM4/idefics-9b-instruct": (
        "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics-9b-instruct"
    ),
    "HuggingFaceM4/idefics-80b-instruct": (
        "https://api-inference.huggingface.co/models/HuggingFaceM4/idefics-80b-instruct"
    ),
    "local/idefics-9b-instruct": (
        "http://35.173.104.196:8080"
    ),
}

SYSTEM_PROMPT = [
    """The following is a conversation between a highly knowledgeable and intelligent visual AI assistant, called Assistant, and a human user, called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about the content of visual inputs. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.
The conversation begins:""",
    """\nUser:""",
    "https://i1.pickpik.com/photos/515/906/355/kittens-cats-pet-cute-preview.jpg",
    "Describe this image.<end_of_utterance>",
    """\nAssistant: Five kittens are standing together in the center of the photograph. From the left to right, there is one orange kitten, two kittens white and black stripes, and two brown kittens with an orange head. They are in bright green grass and it looks like they are moving forward.<end_of_utterance>""",
    "\nUser:How about this image?",
    "https://cdn.pixabay.com/photo/2017/09/25/13/12/puppy-2785074_1280.jpg",
    "Can you describe it too?<end_of_utterance>",
    """\nAssistant: A dog is lying on the floor, looking at the camera. It is looking directly at you. The dog has a white body and brown patches on its face and ears. Its eyes are dark. Its nose is black, and it has long, floppy ears, white paws, long fur, big eyes.<end_of_utterance>""",
    "\nUser: What kind of breed is it?<end_of_utterance>",
    """\nAssistant: It looks like a Cavalier King Charles Spaniel. These breeds are known for their gentle, affectionate nature. Their long ears and big, brown eyes give them a sweet, expressive face. Cavaliers are small dogs weighing between 13 and 20 pounds. They have a silky, medium-length coat that comes in various colors, including black and tan, tricolor, red and white, or black.<end_of_utterance>""",
    "\nUser: What can you tell me about this breed of dogs?<end_of_utterance>",
    """\nAssistant: One specific characteristic of the Cavalier King Charles Spaniel is its friendly and affectionate nature. This breed is known for being extremely sociable and forming strong bonds with their owners. They are often described as "velcro dogs" because they love to be close to their human companions, whether it's sitting on their laps, cuddling on the couch, or simply following them around the house.<end_of_utterance>""",
    "\nUser: ghjkhjabnufs<end_of_utterance>",
    """\nAssistant: That doesn’t seem to be a word. Could you ask me another way?<end_of_utterance>""",
    "\nUser: Do you like Cavalier King Charles Spaniel?<end_of_utterance>",
    """\nAssistant: I do not have personal opinions as I’m just a computer program. However, cavaliers are known for being excellent family pets due to their gentle and patient demeanor, making them great with children and other pets. Their affectionate nature also makes them ideal therapy dogs, as they can provide comfort and emotional support to those in need.<end_of_utterance>""",
    "\nUser: How many dogs do you see in this image?",
    "https://i.dailymail.co.uk/i/pix/2011/07/01/article-2010308-0CD22A8300000578-496_634x414.jpg",
    "<end_of_utterance>",
    """\nAssistant: There is no dogs in this image. The picture shows a tennis player jumping to volley the ball.<end_of_utterance>""",
]

BAN_TOKENS = (  # For documentation puporse. We are not using this list, it is hardcoded inside `idefics_causal_lm.py` inside TGI.
    "<image>;<fake_token_around_image>"
)
EOS_STRINGS = ["<end_of_utterance>", "\nUser:"]
STOP_SUSPECT_LIST = []

#GRADIO_LINK = "https://huggingfacem4-idefics-playground.hf.space"
API_TOKEN = os.getenv("hf_api_token")
IDEFICS_LOGO = "IDEFICS_logo.png"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

PROCESSOR = AutoProcessor.from_pretrained(
    #"HuggingFaceM4/idefics-9b-instruct",
    #token=os.getenv("hf_api_token"),
    "/home/alfred/models/idefics-9b-instruct"
)

BOT_AVATAR = "IDEFICS_logo.png"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# Monkey patch adapted from gradio.components.image.Image - mostly to make the `save` step optional in `pil_to_temp_file`
def hash_bytes(bytes: bytes):
    sha1 = hashlib.sha1()
    sha1.update(bytes)
    return sha1.hexdigest()


def pil_to_temp_file(img: PIL.Image.Image, dir: str = DEFAULT_TEMP_DIR, format: str = "png") -> str:
    """Save a PIL image into a temp file"""
    bytes_data = processing_utils.encode_pil_to_bytes(img, format)
    temp_dir = Path(dir) / hash_bytes(bytes_data)
    temp_dir.mkdir(exist_ok=True, parents=True)
    filename = str(temp_dir / f"image.{format}")
    if not os.path.exists(filename):
        img.save(filename, pnginfo=processing_utils.get_pil_metadata(img))
    return filename


def add_file(file):
    print(file.name)
    return file.name, gr.update(label='Uploaded!')
    

# This is a hack to make pre-computing the default examples work.
# During normal inference, we pass images as url to a local file using the method `gradio_link`
# which allows the tgi server to fetch the local image from the frontend server.
# however, we are building the space (and pre-computing is part of building the space), the frontend is not available
# and won't answer. So tgi server will try to fetch an image that is not available yet, which will result in a timeout error
# because tgi will never be able to return the generation.
# To bypass that, we pass instead the images URLs from the spaces repo.
all_images = glob.glob(f"{os.path.dirname(__file__)}/example_images/*")
DEFAULT_IMAGES_TMP_PATH_TO_URL = {}
for im_path in all_images:
    H = gr.Image(im_path, visible=False, type="filepath")
    print(im_path)
    tmp_filename = H.preprocess(H.value)
    DEFAULT_IMAGES_TMP_PATH_TO_URL[tmp_filename] = f"{os.path.basename(im_path)}"


# Utils to handle the image markdown display logic
def split_str_on_im_markdown(string: str) -> List[str]:
    """
    Extract from a string (typically the user prompt string) the potential images from markdown
    Examples:
    - `User:![](https://favurl.com/chicken_on_money.png)Describe this image.` would become `["User:", "https://favurl.com/chicken_on_money.png", "Describe this image."]`
    - `User:![](/file=/my_temp/chicken_on_money.png)Describe this image.` would become `["User:", "/my_temp/chicken_on_money.png", "Describe this image."]`
    """
    IMAGES_PATTERN = re.compile(r"!\[[^\]]*\]\((.*?)\s*(\"(?:.*[^\"])\")?\s*\)")
    parts = []
    cursor = 0
    for pattern in IMAGES_PATTERN.finditer(string):
        start = pattern.start()
        if start != cursor:
            parts.append(string[cursor:start])
        image_url = pattern.group(1)
        if image_url.startswith("/file="):
            image_url = image_url[6:]  # Remove the 'file=' prefix
        parts.append(image_url)
        cursor = pattern.end()
    if cursor != len(string):
        parts.append(string[cursor:])
    return parts


def is_image(string: str) -> bool:
    """
    There are two ways for images: local image path or url.
    """
    return is_url(string) or string.startswith(DEFAULT_TEMP_DIR)


def is_url(string: str) -> bool:
    """
    Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url
    """
    if " " in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])


def isolate_images_urls(prompt_list: List) -> List:
    """
    Convert a full string prompt to the list format expected by the processor.
    In particular, image urls (as delimited by <fake_token_around_image>) should be their own elements.
    From:
    ```
    [
        "bonjour<fake_token_around_image><image:IMG_URL><fake_token_around_image>hello",
        PIL.Image.Image,
        "Aurevoir",
    ]
    ```
    to:
    ```
    [
        "bonjour",
        IMG_URL,
        "hello",
        PIL.Image.Image,
        "Aurevoir",
    ]
    ```
    """
    linearized_list = []
    for prompt in prompt_list:
        # Prompt can be either a string, or a PIL image
        if isinstance(prompt, PIL.Image.Image):
            linearized_list.append(prompt)
        elif isinstance(prompt, str):
            if "<fake_token_around_image>" not in prompt:
                linearized_list.append(prompt)
            else:
                prompt_splitted = prompt.split("<fake_token_around_image>")
                for ps in prompt_splitted:
                    if ps == "":
                        continue
                    if ps.startswith("<image:"):
                        linearized_list.append(ps[7:-1])
                    else:
                        linearized_list.append(ps)
        else:
            raise TypeError(
                f"Unrecognized type for `prompt`. Got {type(type(prompt))}. Was expecting something in [`str`,"
                " `PIL.Image.Image`]"
            )
    return linearized_list


def fetch_images(url_list: str) -> PIL.Image.Image:
    """Fetching images"""
    return PROCESSOR.image_processor.fetch_images(url_list)


def handle_manual_images_in_user_prompt(user_prompt: str) -> List[str]:
    """
    Handle the case of textually manually inputted images (i.e. the `<fake_token_around_image><image:IMG_URL><fake_token_around_image>`) in the user prompt
    by fetching them, saving them locally and replacing the whole sub-sequence the image local path.
    """
    if "<fake_token_around_image>" in user_prompt:
        splitted_user_prompt = isolate_images_urls([user_prompt])
        resulting_user_prompt = []
        for u_p in splitted_user_prompt:
            if is_url(u_p):
                img = fetch_images([u_p])[0].to(device)
                tmp_file = pil_to_temp_file(img)
                resulting_user_prompt.append(tmp_file)
            else:
                resulting_user_prompt.append(u_p)
        return resulting_user_prompt
    else:
        return [user_prompt]


def gradio_link(img_path: str) -> str:
    url = f"file={img_path}"
    return url


def prompt_list_to_markdown(prompt_list: List[str]) -> str:
    """
    Convert a user prompt in the list format (i.e. elements are either a PIL image or a string) into
    the markdown format that is used for the chatbot history and rendering.
    """
    resulting_string = ""
    for elem in prompt_list:
        if is_image(elem):
            if is_url(elem):
                resulting_string += f"![]({elem})"
            else:
                resulting_string += f"![](/file={elem})"
        else:
            resulting_string += elem
    return resulting_string


def prompt_list_to_tgi_input(prompt_list: List[str]) -> str:
    """
    TGI expects a string that contains both text and images in the image markdown format (i.e. the `![]()` ).
    The images links are parsed on TGI side
    """
    result_string_input = ""
    for elem in prompt_list:
        if is_image(elem):
            if is_url(elem):
                result_string_input += f"![]({elem})"
            else:
                result_string_input += f"![]({gradio_link(img_path=elem)})"
        else:
            result_string_input += elem
    return result_string_input


def remove_spaces_around_token(text: str) -> str:
    pattern = r"\s*(<fake_token_around_image>)\s*"
    replacement = r"\1"
    result = re.sub(pattern, replacement, text)
    return result


# Chatbot utils
def format_user_prompt_with_im_history_and_system_conditioning(
    current_user_prompt_str: str, current_image: Optional[str], history: List[Tuple[str, str]]
) -> Tuple[List[str], List[str]]:
    """
    Produces the resulting list that needs to go inside the processor.
    It handles the potential image box input, the history and the system conditionning.
    """
    resulting_list = copy.deepcopy(SYSTEM_PROMPT)

    # Format history
    for turn in history:
        user_utterance, assistant_utterance = turn
        splitted_user_utterance = split_str_on_im_markdown(user_utterance)

        optional_space = ""
        if not is_image(splitted_user_utterance[0]):
            optional_space = " "
        resulting_list.append(f"\nUser:{optional_space}")
        resulting_list.extend(splitted_user_utterance)
        resulting_list.append(f"<end_of_utterance>\nAssistant: {assistant_utterance}")

    # Format current input
    current_user_prompt_str = remove_spaces_around_token(current_user_prompt_str)
    if current_image is None:
        if "![](" in current_user_prompt_str:
            current_user_prompt_list = split_str_on_im_markdown(current_user_prompt_str)
        else:
            current_user_prompt_list = handle_manual_images_in_user_prompt(current_user_prompt_str)

        optional_space = ""
        if not is_image(current_user_prompt_list[0]):
            # Check if the first element is an image (and more precisely a path to an image)
            optional_space = " "
        resulting_list.append(f"\nUser:{optional_space}")
        resulting_list.extend(current_user_prompt_list)
        resulting_list.append("<end_of_utterance>\nAssistant:")
    else:
        # Choosing to put the image first when the image is inputted through the UI, but this is an arbiratrary choice.
        resulting_list.extend(["\nUser:", current_image, f"{current_user_prompt_str}<end_of_utterance>\nAssistant:"])
        current_user_prompt_list = [current_user_prompt_str]

    return resulting_list, current_user_prompt_list


# dope_callback = gr.CSVLogger()
# problematic_callback = gr.CSVLogger()

textbox = gr.Textbox(
    placeholder="Upload an image and send a message",
    show_label=False,
    # value="Describe the battle against the fierce dragons.",
    visible=True,
    container=False,
    label="Text input",
    scale=6,
)
with gr.Blocks(title="IDEFICS Playground", theme=gr.themes.Base()) as demo:
    gr.HTML("""<h1 align="center">🐶 IDEFICS Playground</h1>""")
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            gr.Image(IDEFICS_LOGO, elem_id="banner-image", show_label=False, show_download_button=False)
        with gr.Column(scale=5):
            gr.HTML("""
                <p>This demo showcases <strong>IDEFICS</strong>, a open-access large visual language model. Like GPT-4, the multimodal model accepts arbitrary sequences of image and text inputs and produces text outputs. IDEFICS can answer questions about images, describe visual content, create stories grounded in multiple images, etc.</p>
                <p>IDEFICS (which stands for <strong>I</strong>mage-aware <strong>D</strong>ecoder <strong>E</strong>nhanced à la <strong>F</strong>lamingo with <strong>I</strong>nterleaved <strong>C</strong>ross-attention<strong>S</strong>) is an open-access reproduction of <a href="https://huggingface.co/papers/2204.14198">Flamingo</a>, a closed-source visual language model developed by Deepmind. IDEFICS was built solely on publicly available data and models. It is currently the only visual language model of this scale (80 billion parameters) that is available in open-access.</p>
                <p>📚 The variants available in this demo were fine-tuned on a mixture of supervised and instruction fine-tuning datasets to make the models more suitable in conversational settings. For more details, we refer to our <a href="https://huggingface.co/blog/idefics">blog post</a>.</p>
                <p>🅿️ <strong>Intended uses:</strong> This demo along with the <a href="https://huggingface.co/models?sort=trending&amp;search=HuggingFaceM4%2Fidefics">supporting models</a> are provided as research artifacts to the community. We detail misuses and out-of-scope uses <a href="https://huggingface.co/HuggingFaceM4/idefics-80b#misuse-and-out-of-scope-use">here</a>.</p>
                <p>⛔️ <strong>Limitations:</strong> The model can produce factually incorrect texts, hallucinate facts (with or without an image) and will struggle with small details in images. While the model will tend to refuse answering questionable user requests, it can produce problematic outputs (including racist, stereotypical, and disrespectful texts), in particular when prompted to do so. We encourage users to read our findings from evaluating the model for potential biases in the <a href="https://huggingface.co/HuggingFaceM4/idefics-80b#bias-evaluation">model card</a>.</p>
            """)

    # with gr.Row():
    #     with gr.Column(scale=2):
    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=MODELS,
            value="HuggingFaceM4/idefics-80b-instruct", #value="local/idefics-9b-instruct",
            interactive=True,
            show_label=False,
            container=False,
            label="Model",
            visible=False,
        )

    imagebox = gr.Image(type="filepath", label="Image input", visible=False)

    with gr.Row():
        # def prefetch_images_in_history(user_prompt_str):
        #     """
        #     Pre-fetch the images that are passed in the chatbot default history.
        #     """
        #     return prompt_list_to_markdown(handle_manual_images_in_user_prompt(user_prompt_str))

        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="IDEFICS",
            visible=True,
            height=750,
            avatar_images=[None, BOT_AVATAR]
            #                 value=[
            #                     [
            #                         (
            #                             prefetch_images_in_history(
            #                                 """Write a funny story including the following characters.
            # There should be drama and most importantly romance.
            # Character 1:<fake_token_around_image><image:https://m.media-amazon.com/images/M/MV5BODg1OTczMWEtNTU3MS00OTUzLThjODEtNDg1MWQwZmExYmFlXkEyXkFqcGdeQWFybm8@._V1_QL75_UX500_CR0,0,500,281_.jpg><fake_token_around_image>
            # Character 2:<fake_token_around_image><image:https://static.wikia.nocookie.net/dreamworks/images/0/09/Shrek_dreamworks.png/revision/latest?cb=20170921225439><fake_token_around_image>
            # Character 3:<fake_token_around_image><image:https://upload.wikimedia.org/wikipedia/en/e/ef/Marsupilami_1.jpg><fake_token_around_image>
            # The plot should take place in the world pictured here:<fake_token_around_image><image:https://www.astronomy.com/wp-content/uploads/sites/2/2021/09/ON1.jpg><fake_token_around_image>"""
            #                             )
            #                         ),
            #                         """Once upon a time, in a world where the sky was a dome and the ground was covered in grass, there lived a beautiful princess named Mulan. She was kind, brave, and always ready to help others. One day, while walking through the forest, she stumbled upon a strange creature. It was a green ogre named Shrek, who was lost and looking for his way back home.
            # Mulan, being the kind-hearted person she was, decided to help Shrek find his way back. As they traveled together, they began to develop a strong bond. Shrek was impressed by Mulan's bravery and determination, while Mulan admired Shrek's loyalty and sense of humor.
            # However, their journey was not without obstacles. They encountered a mischievous cat named Hoboua, who was always causing trouble and getting into fights. Despite his troublemaking ways, Hoboua had a good heart and eventually became a valuable ally to Mulan and Shrek.
            # As they traveled, they faced many challenges, including battling fierce dragons and navigating treacherous mountains. Through it all, Mulan and Shrek grew closer, and their feelings for each other deepened.
            # Finally, they reached Shrek's home, and he was reunited with his family and friends. Mulan, however, was sad to leave him behind. But Shrek had a surprise for her. He had fallen in love with her and wanted to be with her forever.
            # Mulan was overjoyed, and they shared a passionate kiss. From that day on, they lived happily ever after, exploring the world together and facing any challenges that came their way.
            # And so, the story of Mulan and Shrek's romance came to an end, leaving a lasting impression on all who heard it.""",
            #                     ],
            #                 ],
        )

    with gr.Group():
        with gr.Row():
                textbox.render()
                submit_btn = gr.Button(value="▶️ Submit", visible=True)
                clear_btn = gr.ClearButton([textbox, imagebox, chatbot], value="🧹 Clear")
                regenerate_btn = gr.Button(value="🔄 Regenerate", visible=True)
                upload_btn = gr.UploadButton("📁 Upload image", file_types=["image"])
    # with gr.Group():
    #     with gr.Row():
    #         with gr.Column(scale=1, min_width=50):
    #             dope_bttn = gr.Button("Dope🔥")
    #         with gr.Column(scale=1, min_width=50):
    #             problematic_bttn = gr.Button("Problematic😬")

    with gr.Row():
        with gr.Accordion("Advanced settings", open=False, visible=True) as parameter_row:
            max_new_tokens = gr.Slider(
                minimum=8,
                maximum=1024,
                value=512,
                step=1,
                interactive=True,
                label="Maximum number of new tokens to generate",
            )
            repetition_penalty = gr.Slider(
                minimum=0.0,
                maximum=5.0,
                value=1.0,
                step=0.01,
                interactive=True,
                label="Repetition penalty",
                info="1.0 is equivalent to no penalty",
            )
            decoding_strategy = gr.Radio(
                [
                    "Greedy",
                    "Top P Sampling",
                ],
                value="Greedy",
                label="Decoding strategy",
                interactive=True,
                info="Higher values is equivalent to sampling more low-probability tokens.",
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=5.0,
                value=0.4,
                step=0.1,
                interactive=True,
                visible=False,
                label="Sampling temperature",
                info="Higher values will produce more diverse outputs.",
            )
            decoding_strategy.change(
                fn=lambda selection: gr.Slider.update(
                    visible=(
                        selection in ["contrastive_sampling", "beam_sampling", "Top P Sampling", "sampling_top_k"]
                    )
                ),
                inputs=decoding_strategy,
                outputs=temperature,
            )
            top_p = gr.Slider(
                minimum=0.01,
                maximum=0.99,
                value=0.8,
                step=0.01,
                interactive=True,
                visible=False,
                label="Top P",
                info="Higher values is equivalent to sampling more low-probability tokens.",
            )
            decoding_strategy.change(
                fn=lambda selection: gr.Slider.update(visible=(selection in ["Top P Sampling"])),
                inputs=decoding_strategy,
                outputs=top_p,
            )
            gr.Markdown(
                """<p><strong>💡 Pro tip</strong>:<br>
                You can input an arbitrary number of images at arbitrary positions in the same query.<br>
                You will need to input each image with its URL with the syntax <code>&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL&gt;&lt;fake_token_around_image&gt;</code>.<br>
                For example, for two images, you could input <code>TEXT_1&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL_1&gt;&lt;fake_token_around_image&gt;TEXT_2&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL_2&gt;&lt;fake_token_around_image&gt;TEXT_3</code>.<br>
                In the particular case where two images are consecutive, it is not necessary to add an additional separator: <code>&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL_1&gt;&lt;fake_token_around_image&gt;&lt;image:IMAGE_URL_2&gt;&lt;fake_token_around_image&gt;</code>.</p>"""
            )

    def model_inference(
        model_selector,
        user_prompt_str,
        chat_history,
        image,
        decoding_strategy,
        temperature,
        max_new_tokens,
        repetition_penalty,
        top_p,
    ):
        if user_prompt_str.strip() == "" and image is None:
            return "", None, chat_history

        formated_prompt_list, user_prompt_list = format_user_prompt_with_im_history_and_system_conditioning(
            current_user_prompt_str=user_prompt_str.strip(),
            current_image=image,
            history=chat_history,
        )

        client_endpoint = API_PATHS[model_selector]
        client = Client(
            base_url=client_endpoint,
            headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
        )

        # Common parameters to all decoding strategies
        # This documentation is useful to read: https://huggingface.co/docs/transformers/main/en/generation_strategies
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "stop_sequences": EOS_STRINGS,
        }

        assert decoding_strategy in [
            "Greedy",
            "Top P Sampling",
        ]
        if decoding_strategy == "Greedy":
            generation_args["do_sample"] = False
        elif decoding_strategy == "Top P Sampling":
            generation_args["temperature"] = temperature
            generation_args["do_sample"] = True
            generation_args["top_p"] = top_p

        if image is None:
            # Case where there is no image OR the image is passed as `<fake_token_around_image><image:IMAGE_URL><fake_token_around_image>`
            chat_history.append([prompt_list_to_markdown(user_prompt_list), ''])
        else:
            # Case where the image is passed through the Image Box.
            # Convert the image into base64 for both passing it through the chat history and
            # displaying the image inside the same bubble as the text.
            chat_history.append(
                [
                    f"{prompt_list_to_markdown([image] + user_prompt_list)}",
                    '',
                ]
            )

        query = prompt_list_to_tgi_input(formated_prompt_list)
        stream = client.generate_stream(prompt=query, **generation_args)

        acc_text = ""
        for idx, response in enumerate(stream):
            text_token = response.token.text

            if response.details:
                # That's the exit condition
                return

            if text_token in STOP_SUSPECT_LIST:
                acc_text += text_token
                continue

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token.lstrip()

            acc_text += text_token
            last_turn = chat_history.pop(-1)
            last_turn[-1] += acc_text
            if last_turn[-1].endswith("\nUser"):
                # Safeguard: sometimes (rarely), the model won't generate the token `<end_of_utterance>` and will go directly to generating `\nUser:`
                # It will thus stop the generation on `\nUser:`. But when it exits, it will have already generated `\nUser`
                # This post-processing ensures that we don't have an additional `\nUser` wandering around.
                last_turn[-1] = last_turn[-1][:-5]
            chat_history.append(last_turn)
            yield "", None, chat_history
            acc_text = ""

    def process_example(message, image):
        """
        Same as `model_inference` but in greedy mode and with the 80b-instruct.
        Specifically for pre-computing the default examples.
        """
        model_selector="HuggingFaceM4/idefics-80b-instruct" #"local/idefics-9b/instruct"
        user_prompt_str=message
        chat_history=[]
        max_new_tokens=512

        formated_prompt_list, user_prompt_list = format_user_prompt_with_im_history_and_system_conditioning(
            current_user_prompt_str=user_prompt_str.strip(),
            current_image=image,
            history=chat_history,
        )

        client_endpoint = API_PATHS[model_selector]
        client = Client(
            base_url=client_endpoint,
            headers={"x-use-cache": "0", "Authorization": f"Bearer {API_TOKEN}"},
            timeout=240, # Generous time out just in case because we are in greedy. All examples should be computed in less than 30secs with the 80b-instruct.
        )

        # Common parameters to all decoding strategies
        # This documentation is useful to read: https://huggingface.co/docs/transformers/main/en/generation_strategies
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": None,
            "stop_sequences": EOS_STRINGS,
            "do_sample": False,
        }

        if image is None:
            # Case where there is no image OR the image is passed as `<fake_token_around_image><image:IMAGE_URL><fake_token_around_image>`
            chat_history.append([prompt_list_to_markdown(user_prompt_list), ''])
        else:
            # Case where the image is passed through the Image Box.
            # Convert the image into base64 for both passing it through the chat history and
            # displaying the image inside the same bubble as the text.
            chat_history.append(
                [
                    f"{prompt_list_to_markdown([image] + user_prompt_list)}",
                    '',
                ]
            )

        # Hack - see explanation in `DEFAULT_IMAGES_TMP_PATH_TO_URL`
        for idx, i in enumerate(formated_prompt_list):
            if i.startswith(DEFAULT_TEMP_DIR):
                for k, v in DEFAULT_IMAGES_TMP_PATH_TO_URL.items():
                    if k == i:
                        formated_prompt_list[idx] = v
                        break

        query = prompt_list_to_tgi_input(formated_prompt_list)
        generated_text = client.generate(prompt=query, **generation_args).generated_text
        if generated_text.endswith("\nUser"):
            generated_text = generated_text[:-5]

        last_turn = chat_history.pop(-1)
        last_turn[-1] += generated_text
        chat_history.append(last_turn)
        return "", None, chat_history

    textbox.submit(
        fn=model_inference,
        inputs=[
            model_selector,
            textbox,
            chatbot,
            imagebox,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=[textbox, imagebox, chatbot],
    )
    submit_btn.click(
        fn=model_inference,
        inputs=[
            model_selector,
            textbox,
            chatbot,
            imagebox,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=[
            textbox,
            imagebox,
            chatbot,
        ],
    )

    def remove_last_turn(chat_history):
        if len(chat_history) == 0:
            return gr.Update(), gr.Update()
        last_interaction = chat_history[-1]
        chat_history = chat_history[:-1]
        chat_update = gr.update(value=chat_history)
        text_update = gr.update(value=last_interaction[0])
        return chat_update, text_update

    regenerate_btn.click(fn=remove_last_turn, inputs=chatbot, outputs=[chatbot, textbox]).then(
        fn=model_inference,
        inputs=[
            model_selector,
            textbox,
            chatbot,
            imagebox,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=[
            textbox,
            imagebox,
            chatbot,
        ],
    )

    upload_btn.upload(add_file, [upload_btn], [imagebox, upload_btn], queue=False)
    submit_btn.click(lambda : gr.update(label='📁 Upload image', interactive=True), [], upload_btn)
    textbox.submit(lambda : gr.update(label='📁 Upload image', interactive=True), [], upload_btn)
    clear_btn.click(lambda : gr.update(label='📁 Upload image', interactive=True), [], upload_btn)
    
    # Using Flagging for saving dope and problematic examples
    # Dope examples flagging
    # dope_callback.setup(
    #     [
    #         model_selector,
    #         textbox,
    #         chatbot,
    #         imagebox,
    #         decoding_strategy,
    #         temperature,
    #         max_new_tokens,
    #         repetition_penalty,
    #         top_p,
    #     ],
    #     "gradio_dope_data_points",
    # )
    # dope_bttn.click(
    #     lambda *args: dope_callback.flag(args),
    #     [
    #         model_selector,
    #         textbox,
    #         chatbot,
    #         imagebox,
    #         decoding_strategy,
    #         temperature,
    #         max_new_tokens,
    #         repetition_penalty,
    #         top_p,
    #     ],
    #     None,
    #     preprocess=False,
    # )
    # # Problematic examples flagging
    # problematic_callback.setup(
    #     [
    #         model_selector,
    #         textbox,
    #         chatbot,
    #         imagebox,
    #         decoding_strategy,
    #         temperature,
    #         max_new_tokens,
    #         repetition_penalty,
    #         top_p,
    #     ],
    #     "gradio_problematic_data_points",
    # )
    # problematic_bttn.click(
    #     lambda *args: problematic_callback.flag(args),
    #     [
    #         model_selector,
    #         textbox,
    #         chatbot,
    #         imagebox,
    #         decoding_strategy,
    #         temperature,
    #         max_new_tokens,
    #         repetition_penalty,
    #         top_p,
    #     ],
    #     None,
    #     preprocess=False,
    # )

    # gr.Markdown("""## How to use?

    #     There are two ways to provide image inputs:
    #     - Using the image box on the left panel
    #     - Using the inline syntax: `text<fake_token_around_image><image:URL_IMAGE><fake_token_around_image>text`

    #     The second syntax allows inputting an arbitrary number of images.""")

    examples_path = os.path.dirname(__file__)
    #examples_path = "https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/blob/main"
    gr.Examples(
        examples=[
            [
                (
                    "Which famous person does the person in the image look like? Could you craft an engaging narrative"
                    " featuring this character from the image as the main protagonist?"
                ),
                f"{examples_path}/example_images/obama-harry-potter.jpg",
            ],
            [
                "Can you describe the image? Do you think it's real?",
                f"{examples_path}/example_images/rabbit_force.png",
            ],
            ["Explain this meme to me.", f"{examples_path}/example_images/meme_french.jpg"],
            ["Give me a short and easy recipe for this dish.", f"{examples_path}/example_images/recipe_burger.webp"],
            [
                "I want to go somewhere similar to the one in the photo. Give me destinations and travel tips.",
                f"{examples_path}/example_images/travel_tips.jpg",
            ],
            [
                "Can you name the characters in the image and give their French names?",
                f"{examples_path}/example_images/gaulois.png",
            ],
            ## ["Describe this image in detail.", f"{examples_path}/example_images/plant_bulb.webp"],
            #["Write a complete sales ad for this product.", f"{examples_path}/example_images/product_ad.jpg"],
            #[
            #    (
            #        "As an art critic AI assistant, could you describe this painting in details and make a thorough"
            #        " critic?"
            #    ),
            #    f"{examples_path}/example_images/art_critic.png",
            #],
            #[
            #    "Can you tell me a very short story based on this image?",
            #    f"{examples_path}/example_images/chicken_on_money.png",
            #],
            ["Write 3 funny meme texts about this image.", f"{examples_path}/example_images/elon_smoking.jpg"],
            [
                "Who is in this picture? Why do people find it surprising?",
                f"{examples_path}/example_images/pope_doudoune.webp",
            ],
            # ["<fake_token_around_image><image:https://assets.stickpng.com/images/6308b83261b3e2a522f01467.png><fake_token_around_image>Make a poem about the company in the image<fake_token_around_image><image:https://miro.medium.com/v2/resize:fit:1400/0*jvDu2oQreHn63-fJ><fake_token_around_image>organizing the Woodstock of AI event,<fake_token_around_image><image:https://nationaltoday.com/wp-content/uploads/2019/12/national-llama-day-1200x834.jpg><fake_token_around_image>and the fact they brought those to the event.", None],
            ["What are the armed baguettes guarding?", f"{examples_path}/example_images/baguettes_guarding_paris.png"],
            # ["Can you describe the image?", f"{examples_path}/example_images/bear_costume.png"],
            ["What is this animal and why is it unusual?", f"{examples_path}/example_images/blue_dog.png"],
            [
                "What is this object and do you think it is horrifying?",
                f"{examples_path}/example_images/can_horror.png",
            ],
            [
                (
                    "What is this sketch for? How would you make an argument to prove this sketch was made by Picasso"
                    " himself?"
                ),
                f"{examples_path}/example_images/cat_sketch.png",
            ],
            ["Which celebrity does this claymation figure look like?", f"{examples_path}/example_images/kanye.jpg"],
            # [
            #     "Is there a celebrity look-alike in this image? What is happening to the person?",
            #     f"{examples_path}/example_images/ryan-reynolds-borg.jpg",
            # ],
            # ["Can you describe this image in details please?", f"{examples_path}/example_images/dragons_playing.png"],
            #["What can you tell me about the cap in this image?", f"{examples_path}/example_images/ironman_cap.png"],
            #[
            #    "Can you write an advertisement for Coca-Cola based on this image?",
            #    f"{examples_path}/example_images/polar_bear_coke.png",
            #],
            ## [
            ##     "What is the rabbit doing in this image? Do you think this image is real?",
            ##     f"{examples_path}/example_images/rabbit_force.png",
            ## ],
            ## ["What is happening in this image and why is it unusual?", f"{examples_path}/example_images/ramen.png"],
            ## [
            ##     "What I should look most forward to when I visit this place?",
            ##     f"{examples_path}/example_images/tree_fortress.jpg",
            ## ],
            ## ["Who is the person in the image and what is he doing?", f"{examples_path}/example_images/tom-cruise-astronaut-pegasus.jpg"],
            #[
            #    "What is happening in this image? Which famous personality does this person in center looks like?",
            #    f"{examples_path}/example_images/gandhi_selfie.jpg",
            #],
            #[
            #    "What do you think the dog is doing and is it unusual?",
            #    f"{examples_path}/example_images/surfing_dog.jpg",
            #],
        ],
        inputs=[textbox, imagebox],
        outputs=[textbox, imagebox, chatbot],
        fn=process_example,
        cache_examples=True,
        examples_per_page=6,
        label=(
            "Click on any example below to get started.\nFor convenience, the model generations have been"
            " pre-computed with `idefics-9b-instruct`."
        ),
    )

demo.queue(concurrency_count=40, max_size=40)
demo.launch(debug=True, server_name="0.0.0.0", server_port=7863, height=2048, share=False)
