from abc import ABC, abstractmethod
from PIL import Image
import torch
from transformers import (
    BlipProcessor, BlipForQuestionAnswering,
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModelForVision2Seq
)


class VLMWrapper(ABC):
    """Base class for VLM wrappers"""
    
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
    
    @abstractmethod
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image"""
        pass


class BLIPWrapper(VLMWrapper):
    """Wrapper for BLIP VQA model"""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        super().__init__('Salesforce/blip-vqa-base', device)
        
        print(f"Loading BLIP model on {device}...")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(self.model_name).to(device)
        self.model.eval()
        print("✓ BLIP model loaded successfully")
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image"""
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)
        
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer


class BLIP2Wrapper(VLMWrapper):
    """Wrapper for BLIP-2 model"""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        super().__init__('Salesforce/blip2-opt-2.7b', device)
        
        print(f"Loading BLIP-2 model on {device}...")
        print("Note: BLIP-2 is a large model and may take time to load...")
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)
        self.model.eval()
        print("✓ BLIP-2 model loaded successfully")
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image"""
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)
        
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        # Remove the prompt from the answer
        answer = answer.replace(prompt, "").strip()
        return answer


class InstructBLIPWrapper(VLMWrapper):
    """Wrapper for InstructBLIP model"""
    
    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        super().__init__('Salesforce/instructblip-vicuna-7b', device)
        
        print(f"Loading InstructBLIP model on {device}...")
        print("Note: InstructBLIP is a very large model and requires significant memory...")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        ).to(device)
        self.model.eval()
        print("✓ InstructBLIP model loaded successfully")
    
    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer a question about an image"""
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=50)
        
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer


def create_model(model_name: str, device: str = 'auto') -> VLMWrapper:
    """Factory function to create VLM models"""
    
    model_map = {
        'blip': BLIPWrapper,
        'blip2': BLIP2Wrapper,
        'instructblip': InstructBLIPWrapper,
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(model_map.keys())}"
        )
    
    return model_map[model_name.lower()](device)