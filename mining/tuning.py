import os
import uuid

import dsp
from dsp import HFModel
import datasets
import peft
import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

import stance


class Tune:

    def compile(self, student, model_name, model_prompt_template, trainset, valset, lr=3e-2, num_epochs=50, gradient_accumulation_steps=8, log_to_wandb=True, **kwargs):
        
        self.student = student
        self.model_name = model_name
        self.model_prompt_template = model_prompt_template
        self.trainset = trainset
        self.valset = valset
        self.kwargs = kwargs
        self.log_to_wandb = log_to_wandb

        self.signature = student.predictors()[0].signature
        
        all_tasks = 'all_tasks' in kwargs and kwargs['all_tasks']
        if all_tasks:
            self.dataset_name = "all_tasks"
        else:
            self.dataset_name = trainset[0].target_opinion.replace(' ', '_')

        train = [self._ex_to_dict(ex) for ex in trainset]
        if valset:
            val = [self._ex_to_dict(ex) for ex in valset]
            dataset = datasets.Dataset.from_list(train + val).train_test_split(test_size=len(val), shuffle=False)
        else:
            dataset = datasets.DatasetDict({'train': datasets.Dataset.from_list(train)})

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.peft_config = self._get_peft_config()
        
        unique_id = str(uuid.uuid4())[:8]
        checkpoint_name = f"{self.dataset_name}_{self.model_name}_{self.peft_config.peft_type}_{unique_id}".replace(
            "/", "_"
        )
        if self.log_to_wandb:
            wandb.run.config[f"{self.dataset_name}_checkpoint_name"] = checkpoint_name

        self.checkpoint_path = os.path.join(".", "model_checkpoints", checkpoint_name)
        self.text_column = "prompt"
        self.label_column = "completion"

        batch_size = 1

        processed_datasets = dataset.map(
            self._preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )


        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=True)
        self.model = peft.get_peft_model(self.model, self.peft_config)

        if "previous_checkpoint_path" in kwargs:
            sd = torch.load(os.path.join(kwargs["previous_checkpoint_path"], "adapter_model.bin"))
            peft.set_peft_model_state_dict(self.model, sd)

        self.train_dataset = processed_datasets["train"]
        if valset:
            self.eval_dataset = processed_datasets["test"]

        train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True
        )
        if valset:
            eval_dataloader = DataLoader(self.eval_dataset, collate_fn=transformers.default_data_collator, batch_size=batch_size, pin_memory=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(0.1 * num_epochs * len(train_dataloader)),
            num_training_steps=((len(train_dataloader) // gradient_accumulation_steps) * num_epochs),
        )

        self.device = 'cuda'
        self.model = self.model.to(self.device)

        # save the initial model so a checkpoint always exists
        self.model.save_pretrained(self.checkpoint_path)

        # ensure initial performance is equivalent to zero shot performance
        if valset:
            best_score = self._validate_initial_performance(eval_dataloader)
        
        beta = 0.5
        if self.log_to_wandb:
            wandb.run.config['beta_for_fbeta'] = beta

        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                try:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss = loss / gradient_accumulation_steps
                    total_loss += loss.detach().float()
                    loss.backward()

                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        

                    for k, v in batch.items():
                        del v
                    for k, v in outputs.items():
                        del v
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    continue

            train_epoch_loss = total_loss / len(train_dataloader)

            metrics_to_log = {"train_epoch_loss": train_epoch_loss}

            if valset:
                eval_precision, eval_recall, eval_f1, eval_fbeta = self._evaluate(eval_dataloader)

                if eval_fbeta > best_score:
                    best_score = eval_fbeta
                    self.model.save_pretrained(self.checkpoint_path)

                eval_str = f"{eval_f1=} {eval_precision=} {eval_recall=}"
                metrics_to_log.update({"eval_f1": eval_f1, "eval_precision": eval_precision, "eval_recall": eval_recall, "fbeta": eval_fbeta})


            print(f"{epoch=}: {train_epoch_loss=}" + (f" {eval_str}" if valset else ""))
            if self.log_to_wandb:
                if not all_tasks:
                    metrics_to_log = {f"{self.dataset_name}_{k}": v for k, v in metrics_to_log.items()}
                wandb.log(metrics_to_log)

        if valset:
            if os.path.exists(os.path.join(self.checkpoint_path, "adapter_model.bin")):
                sd = torch.load(os.path.join(self.checkpoint_path, "adapter_model.bin"))
                peft.set_peft_model_state_dict(self.model, sd)

        lm = self._define_lm()
        self.student.predictors()[0].lm = lm
        
        return student
    
    def load(self, model_name, checkpoint_path, trainset, classifier, model_prompt_template):
        self.model_name = model_name
        self.model = transformers.AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, use_flash_attention_2=True)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, padding_side="left")

        self.signature = classifier.signature
        self.model_prompt_template = model_prompt_template
        self.trainset = trainset
        sd = torch.load(os.path.join(checkpoint_path, "adapter_model.bin"))
        if 'prompt_embeddings' in sd:
            self.peft_config = self._get_default_peft_config(sd['prompt_embeddings'].shape[0])
        else:
            self.kwargs = {}
            if 'ia3' in list(sd.keys())[0]:
                self.kwargs['method'] = 'ia3'
            elif 'lora' in list(sd.keys())[0]:
                self.kwargs['method'] = 'lora'
            self.peft_config = self._get_peft_config()
        
        self.model = peft.get_peft_model(self.model, self.peft_config)
        self.model.to('cuda')
        
        peft.set_peft_model_state_dict(self.model, sd)
        lm = self._define_lm()
        classifier.lm = lm
        return classifier
        
    def _evaluate(self, eval_dataloader):
        self.model.eval()

        eval_outputs = []
        eval_preds = []
        eval_gold = []

        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # with torch.no_grad():
            #     outputs = model(**batch)
            # loss = outputs.loss
            # eval_loss += loss.detach().float()
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch["input_ids"][:,:-1], attention_mask=batch["attention_mask"][:,:-1], max_new_tokens=10, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id
                )
            output = self.tokenizer.batch_decode(outputs[:,batch['input_ids'].shape[1]-1:].detach().cpu().numpy(), skip_special_tokens=True)[0]
            gold = self.tokenizer.batch_decode(batch['labels'][:,-1:].detach().cpu().numpy(), skip_special_tokens=True)[0]
            eval_outputs.append(output)
            eval_preds.append(stance._parse_opinion_answer(output))
            eval_gold.append(stance._parse_opinion_answer(gold))
        
        eval_precision, eval_recall, eval_f1, eval_fbeta = stance.get_stance_f1_score(eval_gold, eval_preds)

        return eval_precision, eval_recall, eval_f1, eval_fbeta


    def _ex_to_dict(self, ex):
        ex = dict(ex)
        completion = ex.pop(self.signature.fields[-1].output_variable)
        ex = dsp.Example(demos=[], **ex)
        prompt = self.signature(ex).strip()
        prompt = self.model_prompt_template.format(prompt=prompt)
        return dict(prompt=prompt, completion=completion)


class FineTune(Tune):

    def _preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = examples[self.text_column]
        targets = examples[self.label_column]
        model_inputs = self.tokenizer([f"{i} {l}" for i, l in zip(inputs, targets)])
        labels = self.tokenizer(targets, add_special_tokens=False)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] # + [tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids
            labels["input_ids"][i] = [-100] * (len(sample_input_ids) - len(label_input_ids)) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        
        max_length = max(len(x) for x in model_inputs["input_ids"])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            # model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            #     max_length - len(sample_input_ids)
            # ) + sample_input_ids
            # model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            #     "attention_mask"
            # ][i]
            # labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])#[:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])#[:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])#[:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def _get_peft_config(self):
        if self.kwargs['method'] == 'lora':
            peft_config = peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=8, 
                lora_alpha=32, 
                lora_dropout=0.1, 
                bias='lora_only'
            )
        elif self.kwargs['method'] == 'ia3':
            peft_config = peft.IA3Config(
                task_type=peft.TaskType.CAUSAL_LM, 
                target_modules=['k_proj', 'v_proj', 'down_proj'],
                feedforward_modules=['down_proj'],
                inference_mode=False
            )
        else:
            raise ValueError(f"method {self.kwargs['method']} not supported")
        return peft_config
    
    def _validate_initial_performance(self, eval_dataloader):
        eval_precision, eval_recall, eval_f1, eval_fbeta = self._evaluate(eval_dataloader)
        print(f"Initial F1: {eval_f1=}")
        if self.log_to_wandb:
            metrics_to_log = {'eval_precision': eval_precision, 'eval_recall': eval_recall, 'eval_f1': eval_f1}
            if "all_tasks" not in self.dataset_name:
                metrics_to_log = {f"{self.dataset_name}_{k}": v for k, v in metrics_to_log.items()}
            wandb.log(metrics_to_log)
        return eval_f1
    
    def _define_lm(self):
        lm = HFModel(self.model_name, is_client=True)
        lm.model = self.model
        lm.tokenizer = self.tokenizer
        lm.model_prompt_template = self.model_prompt_template
        lm.drop_prompt_from_output = True
        lm.kwargs['temperature'] = 0.0
        lm.is_client = False
        lm.kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        lm.kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        return lm

class PromptTune(Tune):

    def _preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = examples[self.text_column]
        targets = examples[self.label_column]
        model_inputs = self.tokenizer(inputs, add_special_tokens=False)
        labels = self.tokenizer(targets, add_special_tokens=False)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i][self.num_virtual_tokens:]
            label_input_ids = labels["input_ids"][i] # + [tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        
        max_length = max(len(x) for x in model_inputs["input_ids"])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            # model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            #     max_length - len(sample_input_ids)
            # ) + sample_input_ids
            # model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            #     "attention_mask"
            # ][i]
            # labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])#[:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])#[:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])#[:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    
    def _get_peft_config(self):
        instructions_kwargs = {k: v for k, v in self.trainset[0].items() if "{" + k + "}" in self.signature.instructions}
        prompt = self.signature(dsp.Example(demos=[], **instructions_kwargs)).strip() + "\n\n---\n\n" # add dashes at the end as this is not automatically done in the func call but occurs in normal call
        prompt = self.model_prompt_template.split("{prompt}")[0] + prompt
        self.num_virtual_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        peft_config = peft.PromptTuningConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            prompt_tuning_init=peft.PromptTuningInit.TEXT,
            num_virtual_tokens=self.num_virtual_tokens,
            prompt_tuning_init_text=prompt,
            tokenizer_name_or_path=self.model_name,
        )
        return peft_config
    
    def _get_default_peft_config(self, num_virtual_tokens):
        peft_config = peft.PromptTuningConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            prompt_tuning_init=peft.PromptTuningInit.RANDOM,
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=self.model_name,
        )
        return peft_config

    def _validate_initial_performance(self, eval_dataloader):
        test_trainset_input_ids = torch.tensor([self.train_dataset[0]['input_ids'][:-1]]).to(self.device)
        inputs_embeds = self.model.word_embeddings(test_trainset_input_ids)
        prompts = self.model.get_prompt(batch_size=1)
        prompts = prompts.to(inputs_embeds.dtype)
        test_peft_input_embeds = torch.cat((prompts, inputs_embeds), dim=1)

        test_prompt = self.model_prompt_template.format(prompt=self.signature(dsp.Example(demos=[], **self.trainset[0])))
        test_input_ids = self.tokenizer(test_prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
        test_input_embeds = self.model.word_embeddings(test_input_ids)
        # assert test_peft_input_embeds.shape == test_input_embeds.shape
        # assert (test_trainset_input_ids == test_input_ids[:, -test_trainset_input_ids.shape[1]:]).all().cpu().item()
        # assert torch.allclose(test_peft_input_embeds, test_input_embeds)

        eval_precision, eval_recall, eval_f1, eval_fbeta = self._evaluate(eval_dataloader)
        print(f"Initial F1: {eval_f1=}")
        if self.log_to_wandb:
            wandb.log({'eval_precision': eval_precision, 'eval_recall': eval_recall, 'eval_f1': eval_f1})
        return eval_f1
    
    def _define_lm(self):
        self.signature.instructions = ""
        dsp.settings.show_guidelines = False

        lm = HFModel(self.model_name, is_client=True)
        lm.model = self.model
        lm.tokenizer = self.tokenizer
        lm.model_prompt_template = "{prompt}" + self.model_prompt_template.split("{prompt}")[1]
        lm.drop_prompt_from_output = True
        lm.kwargs['temperature'] = 0.0
        lm.is_client = False
        lm.kwargs['pad_token_id'] = self.tokenizer.pad_token_id
        lm.kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        return lm
    

class MultiTaskPromptTune(PromptTune):
    def _get_peft_config(self):
        config_kwargs = {}
        config_kwargs['task_type'] = peft.TaskType.CAUSAL_LM
        config_kwargs['tokenizer_name_or_path'] = self.model_name
        config_kwargs['num_transformer_submodules'] = 1

        instructions_kwargs = {k: v for k, v in self.trainset[0].items() if "{" + k + "}" in self.signature.instructions}
        prompt = self.signature(dsp.Example(demos=[], **instructions_kwargs)).strip() + "\n\n---\n\n" # add dashes at the end as this is not automatically done in the func call but occurs in normal call
        prompt = self.model_prompt_template.split("{prompt}")[0] + prompt
        self.num_virtual_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        if self.kwargs['all_tasks']:
            config_kwargs['prompt_tuning_init'] = peft.MultitaskPromptTuningInit.TEXT
            config_kwargs['num_virtual_tokens'] = self.num_virtual_tokens
            config_kwargs['prompt_tuning_init_text'] = prompt
            config_kwargs['num_tasks'] = len(self.kwargs['task_map'])
        else:
            checkpoint_name = f"all_tasks_{self.model_name}_{self.peft_config.peft_type}_{self.peft_config.task_type}_v1.pt".replace(
                "/", "_"
            )
            checkpoint_path = os.path.join(".", "model_checkpoints", checkpoint_name)

            config_kwargs['prompt_tuning_init'] = peft.MultitaskPromptTuningInit.EXACT_SOURCE_TASK
            config_kwargs['num_virtual_tokens'] = self.num_virtual_tokens
            config_kwargs['prompt_tuning_init_state_dict_path'] = os.path.join(checkpoint_path, "adapter_model.bin")
            config_kwargs['num_tasks'] = 1

        peft_config = peft.MultitaskPromptTuningConfig(**config_kwargs)
        return peft_config
    
    def _validate_initial_performance(self, eval_dataloader):

        eval_precision, eval_recall, eval_f1, eval_fbeta = self._evaluate(eval_dataloader)
        print(f"Initial F1: {eval_f1=}")
        if self.log_to_wandb:
            wandb.log({'eval_precision': eval_precision, 'eval_recall': eval_recall, 'eval_f1': eval_f1})
        return eval_f1
    
    def _ex_to_dict(self, ex):
        ex = dict(ex)
        completion = ex.pop(self.signature.fields[-1].output_variable)
        ex = dsp.Example(demos=[], **ex)
        prompt = self.signature(ex).strip()
        prompt = self.model_prompt_template.format(prompt=prompt)
        task_id = self.kwargs['task_map'][ex.target_opinion]
        return dict(prompt=prompt, completion=completion, task_id=task_id)
    
    def _preprocess_function(self, examples):
        batch_size = len(examples[self.text_column])
        inputs = examples[self.text_column]
        targets = examples[self.label_column]
        model_inputs = self.tokenizer(inputs, add_special_tokens=False)
        labels = self.tokenizer(targets, add_special_tokens=False)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i][self.num_virtual_tokens:]
            label_input_ids = labels["input_ids"][i] # + [tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            task_ids = examples['task_id'][i]
            # model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            #     max_length - len(sample_input_ids)
            # ) + sample_input_ids
            # model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            #     "attention_mask"
            # ][i]
            # labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i])#[:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i])#[:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i])#[:max_length])
            examples['task_id'][i] = torch.tensor(task_ids)
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["task_ids"] = examples['task_id']
        return model_inputs
    
    def _evaluate(self, eval_dataloader):
        self.model.eval()

        eval_outputs = []
        eval_preds = []
        eval_gold = []

        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # with torch.no_grad():
            #     outputs = model(**batch)
            # loss = outputs.loss
            # eval_loss += loss.detach().float()
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch["input_ids"][:,:-1], attention_mask=batch["attention_mask"][:,:-1], task_ids=batch['task_ids'], max_new_tokens=10, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id
                )
            output = self.tokenizer.batch_decode(outputs[:,batch['input_ids'].shape[1]-1:].detach().cpu().numpy(), skip_special_tokens=True)[0]
            gold = self.tokenizer.batch_decode(batch['labels'][:,-1:].detach().cpu().numpy(), skip_special_tokens=True)[0]
            eval_outputs.append(output)
            eval_preds.append(stance._parse_opinion_answer(output))
            eval_gold.append(stance._parse_opinion_answer(gold))
        
        eval_precision, eval_recall, eval_f1, eval_fbeta = stance.get_stance_f1_score(eval_gold, eval_preds)

        return eval_precision, eval_recall, eval_f1, eval_fbeta