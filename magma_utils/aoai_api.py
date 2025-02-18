import os
import base64
import json
import cv2
import time
import numpy as np
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider

class GPTModel:
    def __init__(self, model_name="gpt-4o", resource_name="dl-openai-1") -> None:
        print("Initializing model:", model_name)
        self.client = self._get_client(resource_name=resource_name)
        self.model_name = model_name
    
    def _get_client(self, resource_name="dl-openai-1"):
        endpoint = f"https://{resource_name}.openai.azure.com/"
        print("Endpoint:", endpoint)
        api_version = "2024-02-15-preview"  # Replace with the appropriate API version
        
        # ChainedTokenCredential example borrowed from
        # https://github.com/technology-and-research/msr-azure/blob/main/knowledge-base/how-to/Access-Storage-Without-Keys-in-Azure-ML.md
        # Attribution: AI4Science
        azure_credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                # Exclude other credentials we are not interested in.
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in
                # Azure ML Compute jobs that has the client id of the
                # user-assigned managed identity in it.
                # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication#compute-cluster
                # In case it is not set the ManagedIdentityCredential will
                # default to using the system-assigned managed identity, if any.
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            )
        )
        
        token_provider = get_bearer_token_provider(azure_credential,
            "https://cognitiveservices.azure.com/.default")
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )
        
        return client

    def __load_image(self, img_path):
        '''
        load png images
        '''
        img = cv2.imread(img_path)
        img_encoded_bytes = base64.b64encode(cv2.imencode('.jpg', img)[1])
        img_encoded_str = img_encoded_bytes.decode('utf-8')
        return img_encoded_str

    def get_response(self, image_path, conversation_text):
        # encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        image_tag = self.__load_image(image_path)
        
        # split_idx = conversation_text.find("Now you are given the instruction:")
        # system_text = conversation_text[:split_idx]
        # request_text = conversation_text[split_idx:]
        
        messages=[
            # {
            #     "role": "system",
            #     "content": [
            #         {"type": "text",
            #          "text": system_text},
            #     ],
            # },
            # {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": request_text,
            #         },
            #         {
            #             "type": "image",
            #             "image": image_tag,
            #         },

            #     ]
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_tag
                    },
                    {
                        "type": "text",
                        "text": conversation_text
                    }
                ]
            },
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            result_str = json.loads(completion.to_json())["choices"][0]["message"]["content"] # .strip('```python').strip('```').strip())
        except Exception as e:
            result_str = f"API Error: {str(e)}"
        
        # if error code is 429, it means the API call limit has been reached, sleep for 90s and retry
        while "Error code: 429" in result_str:
            time.sleep(90)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                result_str = json.loads(completion.to_json())["choices"][0]["message"]["content"]
            except Exception as e:
                result_str = f"API Error: {str(e)}"

        return result_str

    def get_response_for_images(self, images, systemp_prompt, conversation_text):
        # encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        base64Frames = []
        # sample at most 5 frames uniformly
        sample_idx = np.linspace(0, len(images)-1, num=5, dtype=int)
        images = [images[i] for i in sample_idx]
        for frame in images:
            # get frame size, frame is a PIL Image
            frame = np.array(frame)
            # convert to bgr
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            height, width, _ = frame.shape
            # resize frame's shortest side to 512 and keep original aspect ratio
            if height < width:
                frame = cv2.resize(frame, (int(width * 512 / height), 512))
            else:
                frame = cv2.resize(frame, (512, int(height * 512 / width)))
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        interleaved_images = []
        for i, frame in enumerate(base64Frames):
            interleaved_images.append({"type":"text", "text": f"image {i}"})
            interleaved_images.append({"type":"image_url", "image_url":{"url":f'data:image/jpg;base64,{frame}'}})

        messages=[
            {
                "role": "system",
                "content": systemp_prompt,                 
            },
            {
                "role": "user", 
                "content": [
                    *interleaved_images, 
                    {
                        "type": "text",
                        "text": conversation_text
                    },                                                                      
                ],
            }            
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,                
            )
            result_str = json.loads(completion.to_json())["choices"][0]["message"]["content"] # .strip('```python').strip('```').strip())
        except Exception as e:
            result_str = f"API Error: {str(e)}"
        
        # if error code is 429, it means the API call limit has been reached, sleep for 90s and retry
        while "Error code: 429" in result_str:
            time.sleep(90)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,                
                )
                result_str = json.loads(completion.to_json())["choices"][0]["message"]["content"]
            except Exception as e:
                result_str = f"API Error: {str(e)}"

        return result_str        