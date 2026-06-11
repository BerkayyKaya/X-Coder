import os
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMEngine:
    def __init__(self, model_path : str = None, tokenizer_path : str = None, quantizaton : bool = False):
        if not model_path or  not tokenizer_path:
            raise ValueError("\n\t[ERROR] Lütfen yüklenecek modelin sistem yollarını kontrol ediniz!")
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if (not self.device == "cuda") and (quantizaton):
            raise Exception("\n\t[ERROR] Sistemde CUDA bulunamadı ancak Kuantizasyon isteniyor. Kuantizasyon yalnızca CUDA için geçerlidir.")
        
        self.quantization = quantizaton
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        print(f"\t[INFO] Model Yükleniyor... {self.model_path} -> {self.device}")
        print(f"\t[INFO] Tokenizer Yükleniyor... {self.tokenizer_path} -> {self.device}")

        quantization_config = None

        if self.quantization:
            print("\n\t[INFO] Kuantizasyon Hazırlanıyor...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = torch.bfloat16
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError("\n\t[ERROR] Modelin dosya yolu bulunamadı!")


        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError("\n\t[ERROR] Tokenizerin dosya yolu bulunamadı!")

        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, quantization_config = quantization_config, device_map = "auto")
            print("\t[INFO] Model Başarılı Bir Şekilde Yüklendi!")
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map = "auto")
            print("\t[INFO] Model Başarılı Bir Şekilde Yüklendi!")

        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        print("\t[INFO] Tokenizer Başarılı Bir Şekilde Yüklendi!")
        

        return model, tokenizer
    
    def generate_response(self, messages : list, max_new_tokens : int = 150) -> str:
    
        formatted_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize = False)
        inputs = self.tokenizer(formatted_prompt, return_tensors = "pt").to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens = max_new_tokens, temperature = 0.1, do_sample = True, eos_token_id = self.tokenizer.eos_token_id)

        input_length = inputs["input_ids"].shape[1] # modele giren token sayısı
        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens = True)

        return response.strip()

    def unload_model(self):
        print(f"\n\t[INFO] Model Bellekten Temizleniyor... {self.model_path.split('/')[-1]}")
        del self.model
        del self.tokenizer

        gc.collect()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache() # GPU Belleğini boşa çıkar
        print(f"\n\t[INFO] Model Bellekten Temizlendi!")