import os
import torch
import gc

class LLMEngine:
    def __init__(self, model_path : str = None, tokenizer_path : str = None, backend : str = None, **kwargs):
        if not backend:
            raise TypeError("\n\t[ERROR] Lütfen yüklenecek modelin backendini belirtiniz!")
        if not model_path or  not tokenizer_path:
            raise TypeError("\n\t[ERROR] Lütfen yüklenecek modelin sistem yollarını kontrol ediniz!")
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backend = backend.lower()
        self.kwargs = kwargs

        self.quantization = kwargs.get("quantization")
        if (not self.device == "cuda") and (self.quantization):
            raise Exception("\n\t[ERROR] Sistemde CUDA bulunamadı ancak Kuantizasyon isteniyor. Kuantizasyon yalnızca CUDA için geçerlidir.")
        
        self.model, self.tokenizer = self._load_model(backend = self.backend)

    def _load_model(self, backend : str,):
        valid_backend_val = ["transformers", "llama-cpp", "vllm"]
        if backend == "transformers":
            return self._load_with_transformers()
        elif backend == "llama-cpp":
            return self._load_with_llama()
        elif backend == "vllm":
            pass
        else:
            raise ValueError(f"\n\t[ERROR] Bilinmeyen backend değeri: {backend}\n\tİzin verilen backend değerleri: {valid_backend_val}")

    def generate_response(self, messages: list, max_new_tokens : int = 150, **kwargs) -> str:
        if self.backend == "transformers":
            return self._generate_with_transformers(messages = messages, max_new_tokens = max_new_tokens, **kwargs)
        elif self.backend == "llama-cpp":
            return self._generate_with_llama(messages = messages, max_new_tokens = max_new_tokens, **kwargs)


    #region MODEL LOAD FUNCTIONS

    def _load_with_transformers(self):
        # Import kontrolleri
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"\n\t[ERROR] Sistemde transformers kütüphanesi bulunamadı! "
                                      "\n\tLütfen ilgili kütüphaneyi yükleyiniz:\n\tpip install transformers")
        except ImportError:
            raise ImportError(f"\n\t[ERROR] Sistemde transformers kütüphanesi bulundu ancak modülleri eksik veya hatalı olabilir." 
                              "\n\tLütfen ilgili kütüphaneyi güncelleyin:\n\tpip install --upgrade transformers")
        
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
    
    def _load_with_llama(self):
        try:
            from llama_cpp import Llama
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f"\n\t[ERROR] Sistemde llama_cpp kütüphanesi bulunamadı! "
                                      "Lütfen ilgili kütüphaneyi yükleyiniz:\n\tpip install llama-cpp-python")
        except ImportError:
            raise ImportError(f"\n\t[ERROR] Sistemde llama_cpp kütüphanesi var ancak modülleri eksik veya hatalı olabilir "
                              "\n\tLütfen ilgili kütüphaneyi güncelleyin:\n\tpip install --upgrade llama-cpp-python")
        
        n_gpu_layers = self.kwargs.get("n_gpu_layers", -1) # eğer belirtilmemişse komple gpu'ya yükle
        n_ctx = self.kwargs.get("n_ctx", 4096) # belirtilmemişse context_size değerini 4096 olarak ayarla
        
        print(f"\t[INFO] Model Yükleniyor... {self.model_path.split('/')[-1]}")
        model = Llama(
            model_path = self.model_path,
            n_gpu_layers = n_gpu_layers,
            n_ctx = n_ctx,
            verbose = False
        )

        print(f"\n\t[INFO] Model Başarılı Bir Şekilde Yüklendi!")
        return model, None # llama_cpp tokenizeri kendi içinde yönetiyor o yüzden tokenizere gerek yok
    
    #endregion

    #region RESPONSE GENERATOR FUNCTIONS

    def _generate_with_transformers(self, messages : list, max_new_tokens : int = 150, **kwargs) -> str:
    
        formatted_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True, tokenize = False)
        inputs = self.tokenizer(formatted_prompt, return_tensors = "pt").to(self.device)

        outputs = self.model.generate(
            **inputs, max_new_tokens = max_new_tokens,  
            eos_token_id = self.tokenizer.eos_token_id,
            **kwargs
        )

        input_length = inputs["input_ids"].shape[1] # modele giren token sayısı
        generated_tokens = outputs[0][input_length:]

        response = self.tokenizer.decode(generated_tokens, skip_special_tokens = True)

        return response.strip()


    def _generate_with_llama(self, messages : list, max_new_tokens : int = 150, **kwargs):
        response = self.model.create_chat_completion(
            messages = messages,
            max_tokens = max_new_tokens,
            stream = False,
            **kwargs
        )

        return response["choices"][0]["message"]["content"].strip() # dönen json formatındaki response değerinden sadece içeriği alıyoruz

    #endregion

    def unload_model(self):
        print(f"\n\t[INFO] Model Bellekten Temizleniyor... {self.model_path.split('/')[-1]}")
        del self.model
        del self.tokenizer

        gc.collect()

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache() # GPU Belleğini boşa çıkar
        print(f"\n\t[INFO] Model Bellekten Temizlendi!")