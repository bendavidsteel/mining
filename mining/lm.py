from outlines import models as outline_models
import outlines.text.generate as outlines_generate
import transformers
import torch

class AutoRegressiveLanguageModel:
    def __init__(self, model_name, system_prompt=None, use_regex=False, model_initial_kwargs={}, tokenizer_initial_kwargs={}, generate_kwargs={}, tokenizer_kwargs={}):
        self.device = 'cuda'
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.use_regex = use_regex

        model_kwargs = {'torch_dtype': torch.float16, 'device_map': 'auto'}
        model_kwargs.update(model_initial_kwargs)

        tokenizer_initial_kwargs.update({'use_fast': True})

        if self.use_regex:
            self.guided_model = outline_models.transformers(
                self.model_name, 
                device='cuda', 
                model_kwargs=model_kwargs, 
                tokenizer_kwargs=tokenizer_kwargs
            )
        else:
            self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, **tokenizer_initial_kwargs)

        self.generate_kwargs = {}
        self.generate_kwargs.update(generate_kwargs)
        self.tokenizer_generate_kwargs = {'return_tensors': 'pt', 'add_special_tokens': False, 'padding': True, 'truncation': True, 'max_length':1024}
        self.tokenizer_generate_kwargs.update(tokenizer_kwargs)

    def _format(self, prompt):
        raise NotImplementedError()
    
    def _unformat(self, decoded, prompt):
        raise NotImplementedError()
    
    def __call__(self, prompts, regex=None):
        prompts = [self._format(prompt) for prompt in prompts]
        if self.use_regex:
            decoded = []
            regex_model = outlines_generate.regex(self.guided_model, regex)
            for prompt in prompts:
                decoded.append(regex_model(prompt))

        else:
            self.tokenizer_generate_kwargs['max_length'] = max([len(p) for p in prompts])
            encodes = self.tokenizer(prompts, **self.tokenizer_generate_kwargs)
            model_inputs = encodes.to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(**model_inputs, **self.generate_kwargs)
            decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded = [self._unformat(d, p) for d, p in zip(decoded, prompts)]
        return decoded

class Mistral(AutoRegressiveLanguageModel):
    def __init__(self):
        tokenizer_initial_kwargs = {'use_fast': True, 'padding_side': 'left', 'add_eos_token': True}
        model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
        
        generate_kwargs = {'max_new_tokens': 1024, 'do_sample': True, 'pad_token_id': self.tokenizer.eos_token_id}
        tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length':1024}

        super().__init__(model_name, tokenizer_initial_kwargs=tokenizer_initial_kwargs, generate_kwargs=generate_kwargs, tokenizer_kwargs=tokenizer_kwargs)
        
        # self.tokenizer.pad_token = self.tokenizer.bos_token
        # self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
        # self.model.config.pad_token_id = self.model.config.bos_token_id

        # example
        # text = "<s>[INST] What is your favourite condiment? [/INST]"
        # "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
        # "[INST] Do you have mayonnaise recipes? [/INST]"


    def _format(self, prompt):
        return f"<s>[INST] {prompt} [/INST]"
    
    def _unformat(self, decoded, prompt):
        return decoded.split("[/INST]")[1]
    
class Zephyr(AutoRegressiveLanguageModel):
    def __init__(self):
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
        # self.model = transformers.AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")#, use_flash_attention_2=True)
        model_name = 'HuggingFaceH4/zephyr-7b-beta'
        system_prompt = "You are an expert text annotator, who thinks through problems step by step."
        super().__init__(model_name, system_prompt=system_prompt)
        self.generate_kwargs = {'max_new_tokens': 1024, 'do_sample': True, 'temperature': 0.7, 'top_k': 50, 'top_p': 0.95, 'pad_token_id': self.tokenizer.eos_token_id}

    def _format(self, prompt):
        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": prompt},
        ]
        if self.use_regex:
            tokenizer = self.guided_model.tokenizer.tokenizer
        else:
            tokenizer = self.tokenizer

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def _unformat(self, decoded, prompt):
        return decoded[len(prompt):]
        
def main():
    zephyr = Zephyr()
    res = zephyr(["What is your favourite condiment?"])
    print(res)

    zephyr = Zephyr()
    res = zephyr(["What is your favourite condiment?", "What is your favourite food?"])
    print(res)
    zephyr = None

    mistral = Mistral()
    res = mistral(["What is your favourite condiment?"])
    print(res)

    mistral = Mistral()
    res = mistral(["What is your favourite condiment?", "What is your favourite food?"])
    print(res)

    mistral = None


if __name__ == '__main__':
    main()
