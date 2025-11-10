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


class FlanWrapper(VLMWrapper):
    """Text-only Flan-T5 small wrapper (lazy loads). Image is ignored."""
    def __init__(self, device: str = 'auto', model_name: str = 'google/flan-t5-small'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__(model_name, device)
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except Exception as e:
            raise ImportError('transformers is required for FlanWrapper') from e

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._loaded = True

    def answer_question(self, image: Image.Image, question: str) -> str:
        # image is ignored for this text-only baseline
        self._ensure_loaded()
        inputs = self.tokenizer(question, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=64)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class CLIPWrapper(VLMWrapper):
    """CLIP wrapper returning a scalar similarity string. Lazy-loading to avoid downloads at import time."""
    def __init__(self, device: str = 'auto', model_name: str = 'openai/clip-vit-base-patch32'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__(model_name, device)
        self.processor = None
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from transformers import CLIPProcessor, CLIPModel
        except Exception as e:
            raise ImportError('transformers with CLIP support is required for CLIPWrapper') from e

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._loaded = True

    def answer_question(self, image: Image.Image, question: str) -> str:
        self._ensure_loaded()
        inputs = self.processor(text=[question], images=image, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            # get_image_features and get_text_features are available on CLIPModel
            image_inputs = {k: v for k, v in inputs.items() if k.startswith('pixel_values')}
            text_inputs = {k: v for k, v in inputs.items() if k in ('input_ids', 'attention_mask')}
            image_emb = self.model.get_image_features(**image_inputs)
            text_emb = self.model.get_text_features(**text_inputs)

        image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
        sim = (image_emb @ text_emb.T).squeeze().item()
        return f"similarity:{sim:.4f}"


class Vision2SeqWrapper(VLMWrapper):
    """Generic vision->text wrapper using AutoProcessor + AutoModelForVision2Seq (lazy)."""
    def __init__(self, device: str = 'auto', model_name: str = 'nlpconnect/vit-gpt2-image-captioning'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__(model_name, device)
        self.processor = None
        self.model = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
        except Exception as e:
            raise ImportError('transformers with Vision2Seq support is required for Vision2SeqWrapper') from e

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._loaded = True

    def answer_question(self, image: Image.Image, question: str) -> str:
        # Many vision->seq models accept both image and text; use processor to build inputs
        self._ensure_loaded()
        inputs = self.processor(images=image, text=question, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=64)
        try:
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            # fallback: return token ids as string
            return out[0].cpu().tolist().__repr__()


def create_model(model_name: str, device: str = 'auto') -> VLMWrapper:
    """Factory function to create VLM models"""
    
    model_map = {
        'blip': BLIPWrapper,
        'blip2': BLIP2Wrapper,
        'instructblip': InstructBLIPWrapper,
        'flan': FlanWrapper,
        'clip': CLIPWrapper,
        'vision2seq': Vision2SeqWrapper,
    }

    key = model_name.lower()
    if key not in model_map:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {', '.join(model_map.keys())}"
        )

    return model_map[key](device)