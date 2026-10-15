import torch
import numpy as np
from PIL import Image
import io
import base64
import requests
import json
import copy
import subprocess
import sys
import os

# =================================================================================
# Helper functions and classes
# =================================================================================

class Message:
    """A simple wrapper for the message list to be passed between nodes."""
    def __init__(self, messages=None):
        self.messages = messages if messages is not None else []
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    def get_messages(self):
        return self.messages

# =================================================================================
# Node 1: API Configuration
# =================================================================================

class LMS_APIConfig:
    """⚙️ LM Studio API Configuration Node"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "api_address": ("STRING", {"default": "http://127.0.0.1:1234/v1"}), "api_key": ("STRING", {"default": "lm-studio"})}}
    RETURN_TYPES = ("API_CONFIG",)
    FUNCTION = "configure"
    CATEGORY = "LM Studio Tools"
    def configure(self, api_address, api_key):
        return ({"address": api_address, "key": api_key},)

# =================================================================================
# Node 2: Model Selector
# =================================================================================

class LMS_SelectModel:
    """🏷️ Select Model Node"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_identifier": ("STRING", {"multiline": False, "default": "gemma-2b-it-gguf"})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "select_model"
    CATEGORY = "LM Studio Tools"
    def select_model(self, model_identifier):
        if not model_identifier or not model_identifier.strip():
             raise ValueError("Model Identifier cannot be empty.")
        print(f"LM Studio Tools: Model identifier set to '{model_identifier}'.")
        return (model_identifier,)

# =================================================================================
# Node 3: System Prompt
# =================================================================================

class LMS_SystemPrompt:
    """🧠 System Prompt Node"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"system_prompt": ("STRING", {"multiline": True, "default": "You are a helpful assistant."})}}
    RETURN_TYPES = ("MESSAGE",)
    FUNCTION = "create_system_prompt"
    CATEGORY = "LM Studio Tools"
    def create_system_prompt(self, system_prompt):
        message_instance = Message()
        if system_prompt.strip():
            message_instance.add_message("system", system_prompt)
        return (message_instance,)

# =================================================================================
# Node 4: User Prompt (REVISED FOR MULTIPLE IMAGES)
# =================================================================================

class LMS_UserPrompt:
    """
    👤 User Prompt Node
    Adds a user message to the conversation. Now supports multiple images.
    Connect a batch of images to the 'image' input.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"user_prompt": ("STRING", {"multiline": True, "default": ""})},
            "optional": {"message_in": ("MESSAGE",), "image": ("IMAGE",)}
        }
    RETURN_TYPES = ("MESSAGE",)
    FUNCTION = "add_user_prompt"
    CATEGORY = "LM Studio Tools"
    
    def add_user_prompt(self, user_prompt, message_in=None, image=None):
        new_message_instance = Message(messages=copy.deepcopy(message_in.get_messages())) if message_in else Message()
        
        final_content = None

        # Case 1: Images are provided (single or multiple)
        if image is not None:
            content_parts = []
            
            # First, add the text part to the list
            if user_prompt.strip():
                content_parts.append({"type": "text", "text": user_prompt})

            # The 'image' input is a batch tensor (N, H, W, C)
            # Iterate through each image in the batch
            for i in range(image.shape[0]):
                single_image_tensor = image[i]
                
                # Convert tensor to PIL Image
                img_np = 255. * single_image_tensor.cpu().numpy()
                img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                
                # Encode image to base64 string
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                image_url = f"data:image/png;base64,{img_base64}"
                
                # Append the image part to the content list
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            
            final_content = content_parts
        
        # Case 2: No images are provided
        else:
            if user_prompt.strip():
                final_content = user_prompt

        # Add the constructed message to the conversation history
        if final_content:
            new_message_instance.add_message("user", final_content)
            
        return (new_message_instance,)

# =================================================================================
# Node 5: LM Studio Request
# =================================================================================

class LMS_Request:
    """📡 LM Studio Request Node"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_config": ("API_CONFIG",), "model": ("STRING", {"forceInput": True}), "message": ("MESSAGE",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "context_length": ("INT", {"default": 4096, "min": 10, "max": 100000, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"上一步": ("*",),}
        }
    RETURN_TYPES = ("MESSAGE", "STRING",)
    RETURN_NAMES = ("message_out", "output_text",)
    FUNCTION = "send_request"
    CATEGORY = "LM Studio Tools"
    
    def send_request(self, api_config, model, message, seed, context_length, temperature, top_p, 上一步=None):
        api_address = api_config.get("address")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_config.get('key')}"}
        payload = {
            "model": model, "messages": message.get_messages(), "max_tokens": context_length, 
            "temperature": temperature, "top_p": top_p, "stream": False
        }
        if seed != -1: payload["seed"] = seed

        print(f"LM Studio Tools: Sending request to {api_address} for model '{model}'...")
        try:
            response = requests.post(f"{api_address}/chat/completions", headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            assistant_reply = response.json()['choices'][0]['message']['content']
            print("LM Studio Tools: Request successful.")
            new_message_instance = Message(messages=copy.deepcopy(message.get_messages()))
            new_message_instance.add_message("assistant", assistant_reply)
            return (new_message_instance, assistant_reply,)
        except requests.exceptions.RequestException as e:
            error_message = f"API Request Failed: {e}"
            print(error_message)
            return (message, error_message,)

# =================================================================================
# Node 6: Unload Model
# =================================================================================

class LMS_UnloadModel:
    """🗑️ Unload All Models"""
    @classmethod
    def INPUT_TYPES(s):
        return {"optional": {"message_in": ("MESSAGE",),"string_in": ("STRING", {"forceInput": True}),}}
    RETURN_TYPES = ("MESSAGE", "STRING",)
    RETURN_NAMES = ("message_out", "string_out",)
    FUNCTION = "unload"
    CATEGORY = "LM Studio Tools"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # Unloading is a side effect, so never let ComfyUI serve this node from cache.
        return float("nan")

    def unload(self, message_in=None, string_in=""):
        command = "lms.exe" if sys.platform == "win32" else "lms"
        full_command = [command, "unload", "--all"]
        try:
            print(f"LM Studio Tools: Attempting to unload all models using command: '{' '.join(full_command)}'")
            result = subprocess.run(full_command, check=True, text=True, capture_output=True)
            output_log = result.stdout.strip() or "Successfully unloaded all models."
            print(f"LM Studio Tools: Command executed successfully. -> LMS CLI Output: {output_log}")
        except FileNotFoundError:
            print(f"LM Studio Tools: Error: The '{command}' command was not found.")
        except subprocess.CalledProcessError as e:
            print(f"LM Studio Tools: Error executing command: {e}\nStderr: {e.stderr.strip()}")
        except Exception as e:
            print(f"LM Studio Tools: An unexpected error occurred: {e}")
        return (message_in, string_in,)

# =================================================================================
# Node 7: Get Assistant Message by Index
# =================================================================================

class LMS_GetAssistantMessage:
    """📌 Get Assistant Message"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"message": ("MESSAGE",), "index": ("INT", {"default": -1})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "get_message"
    CATEGORY = "LM Studio Tools"
    def get_message(self, message, index):
        assistant_messages = [m for m in message.get_messages() if m.get('role') == 'assistant']
        if not assistant_messages: return ("",)
        if index == -1: return (assistant_messages[-1].get('content', ''),)
        try:
            return (assistant_messages[index].get('content', ''),)
        except IndexError:
            return ("",)

# =================================================================================
# Node Mappings
# =================================================================================

NODE_CLASS_MAPPINGS = {
    "LMS_APIConfig": LMS_APIConfig, "LMS_SelectModel": LMS_SelectModel,
    "LMS_SystemPrompt": LMS_SystemPrompt, "LMS_UserPrompt": LMS_UserPrompt,
    "LMS_Request": LMS_Request, "LMS_UnloadModel": LMS_UnloadModel,
    "LMS_GetAssistantMessage": LMS_GetAssistantMessage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LMS_APIConfig": "⚙️ API Config", "LMS_SelectModel": "🏷️ Select Model",
    "LMS_SystemPrompt": "🧠 System Prompt", "LMS_UserPrompt": "👤 User Prompt",
    "LMS_Request": "📡 LMStudio Request", "LMS_UnloadModel": "🗑️ Unload All Models",
    "LMS_GetAssistantMessage": "📌 Get Assistant Message",
}