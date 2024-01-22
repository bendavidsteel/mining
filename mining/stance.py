import os
import random
import re

import dspy
from dspy import dsp
from dspy.datasets.dataset import Dataset as DSPyDataset
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithOptuna, BootstrapFewShotWithRandomSearch, LabeledFewShot, BootstrapFinetune
from sklearn import metrics as sk_metrics
import torch

import lm
import tuning

PROMPT_RESPONSES = [
    {
        'name': 'choice',
        'regex': r"(favor|neutral|against)"
    },
    {
        'name': 'thought',
        'regex': r"Reason: [^\.]*\. Stance: (favor|neutral|against)"
    }
]

TARGET_EXPLANATIONS = {
    'vaccine mandates': "Any laws or policies enforcing the use of vaccines, including in specific communities like companies or the army. Includes creating rewards for vaccine use, or creating disincentives for lack of vaccination. Not laws concerning the rest of COVID policy i.e. lockdowns. A personal opinion on vaccines in general counts as neutral on vaccine mandates. But an opinion that others should get vaccines indicates support for vaccine mandates.",
    'renter protections': "Any laws protecting renters - laws limiting landlord actions, capping rent increases, eviction laws, etc.",
    'liberals': "Opinion on the Liberal Party of Canada Otherwise known as the LPC, or Liberals. Opinion on an official party representative counts as an opinion on the party, i.e. opinions on Justin Trudeau count towards opinion for Liberals. Liking or disliking another party doesn't imply disliking or liking the party in question.",
    'conservatives': "Opinion on the Conservative Party of Canada. Otherwise known as the CPC, or Conservatives. Opinion on an official party representative counts as an opinion on the party, i.e. opinions on Pierre Poilievre or Erin O'Toole count towards opinion for Conservatives. Liking or disliking another party doesn't imply disliking or liking the party in question.",
    'ndp': "Opinion on the NDP Party of Canada. Opinion on an official party representative counts as an opinion on the party, i.e. opinions on Jagmeet Singh count towards opinion for NDP. Liking or disliking another party doesn't imply disliking or liking the party in question.",
    'gun control': "Opinion on regulations on weapons, including any weapon ban - assault rifles, handguns, or any restrictions on gun ownership.",
    'drug decriminalization': "Includes opinions on harm reduction measures - i.e. safe spaces to use, government provided supply - anywhere where the government is adding legal situations to do normally illegal drugs",
    'liberal immigration policy': "Opinion on more open, liberal immigration policy, indicating there should be more immigrants, as opposed to a more restrictive immigration policy, implying there should be fewer immigrants. Views on individual immigrants don't count - just whether the government should allow more or restrict immigration.",
    'canadian aid to ukraine': "Opinion on any government aid to Ukraine from Canada, including financial, arms, or in the form of troops. Individual private citizen volunteers don't count.",
    'funding the cbc': "Opinion on any government funding to the CBC, including any funding cuts or increases.",
    'french language laws': "Favor/against laws requiring that people speak French/must attend French schooling/other laws that promote the use of French or limit the use of English in order to increase the use of French"
}

TARGET_NAMES = {
    'vaccine mandates': "vaccine mandates",
    'renter protections': "renter protections",
    'liberals': "the Liberal Party",
    'conservatives': "the Conservative Party",
    'ndp': "the NDP Party",
    'gun control': "gun control",
    'drug decriminalization': "drug decriminalization",
    'liberal immigration policy': "open immigration policy",
    'canadian aid to ukraine': "aid to Ukraine",
    'funding the cbc': "funding the CBC",
    'french language laws': "French language laws"
}

def map_row(r):
    d = {}
    if r[0] is not None:
        d['comment'] = r[0]

    if r[1] is not None:
        d['parent_comment'] = r[1]
    else:
        d['parent_comment'] = ""

    if r[2] is not None:
        d['post'] = r[2]
    else:
        d['post'] = ""

    if len(r) > 3:
        d['gold_stance'] = r[3]

    return d

def get_prompt_template(is_comment, is_replying):
    if is_comment and is_replying:
        prompt = """Given a post: "{post}", and a parent comment replying to that post: "{parent_comment}", what is the attitude of this replying comment: "{comment}" to the target "{target}"?"""
    if is_comment and not is_replying:
        prompt = """Given a post: "{post}", what is the attitude of this comment: "{comment}" to the target "{target}"?"""
    if not is_comment:
        prompt = """What is the attitude of this post: "{post}" to the target "{target}"?"""

    # prompt += """ Respond only with one of the following: strongly against, against, neutral, favor, strongly favor: """
    prompt += """ Respond only with one of the following: against, neutral, favor: """
    return prompt


def _extract_example(field_names, example, raw_pred):
    example = dsp.Example(example)

    raw_pred = raw_pred.strip()

    fields = [
        {
            'input_variable': f,
            'output_variable': f,
            'name': f.replace('_', ' ').title() + ':'
        }
        for f in field_names
    ]

    idx = 0
    while idx < len(fields):
        if (
            fields[idx]['input_variable'] not in example
            or example[fields[idx]['input_variable']] is None
        ):
            break
        idx += 1

    idx = min(idx, len(fields) - 1)
    while raw_pred != "" and idx < len(fields):
        if idx < len(fields) - 1:
            next_field_name = "\n" + fields[idx + 1]['name']
            offset = raw_pred.find(next_field_name)

            if offset >= 0:
                example[fields[idx]['output_variable']] = raw_pred[:offset].strip().split('---')[0].strip()
                raw_pred = raw_pred[offset + len(next_field_name) :].strip().split('---')[0].strip()()

                idx += 1
            else:
                example[fields[idx]['output_variable']] = raw_pred.strip().split('---')[0].strip()

                raw_pred = ""
                idx += 1
                break

        else:
            assert idx == len(fields) - 1, (idx, len(fields))
            example[fields[idx]['output_variable']] = raw_pred.strip().split('---')[0].strip()

            break

    return example
    

class CommentStanceDetectionSignature(dspy.Signature):
    """Predict the stance of the comment towards the target issue.
    If the comment is directly or indirectly agreeing with the target issue, or opposing or critizing something opposed to the target issue, then the opinion should be agree.
    If the comment is directly or indirectly disagreeing with the target issue, or opposing or critizing something in favor of the target issue, then the opinion should be disagree.
    If the comment is discussing something irrelevant to the target, then the opinion should be neutral"""

    target_opinion = dspy.InputField(desc="""The target issue that is either being agreed with, neutral, or disagreed with.""", prefix="Target Issue: ")
    target_explanation = dspy.InputField(desc="""An explanation of the target issue, may be useful in determining the opinion of the comment.""")
    post = dspy.InputField(desc="""The post being commented on, may be useful in determining what the comment is discussing.""")
    parent_comment = dspy.InputField(desc="""The parent comment being replied to, may be useful in determining the context of the comment.""")
    comment = dspy.InputField(desc="""The comment to determine the opinion of.""")
    opinion = dspy.OutputField(desc="""${agree, disagree, or neutral}""", prefix="Opinion: Therefore the opinion of the comment is ")

    def extract(example, raw_pred):
        field_names = ['target_opinion', 'target_explanation', 'post', 'parent_comment', 'comment', 'opinion']
        return _extract_example(field_names, example, raw_pred)
    
class CommentStanceDetectionTemplateSignature(dspy.Signature):
    """Predict the stance of the comment towards {target_opinion}. Here is an explanation of what we mean by {target_opinion}: {target_explanation}
    If the comment is directly or indirectly in favor of {target_opinion}, or opposing or critizing something opposed to {target_opinion}, then the stance should be favor.
    If the comment is directly or indirectly against {target_opinion}, or opposing or critizing something in favor of {target_opinion}, then the stance should be against.
    If the comment is discussing something irrelevant to {target_opinion}, or if it is unclear what the stance is, then the stance should be neutral."""

    post = dspy.InputField(desc="""The post being commented on, may be useful in determining what the comment is discussing.""")
    parent_comment = dspy.InputField(desc="""The parent comment being replied to, may be useful in determining the context of the comment.""")
    comment = dspy.InputField(desc="""The comment to determine the opinion of.""")
    opinion = dspy.OutputField(desc="""${favor, neutral, or against}""", prefix="Stance: The stance of the comment is ")

    def __call__(self, *args, **kwargs):
        return ""

    def extract(example, raw_pred):
        field_names = ['target_opinion', 'target_explanation', 'post', 'parent_comment', 'comment', 'opinion']
        return _extract_example(field_names, example, raw_pred)

class TwoStepCommentStanceDetectionSignature(dspy.Signature):
    """Determine the opinion of the comment towards the target issue.
    First determine if there is clearly an opinion specifically about the target present in the comment. 
    If the comment is clearly expressing an opinion specifically about the target, answer yes.
    If the comment is discussing something different from the target, or if it is not clear what the opinion is, answer no.
    Then determine if the opinion is agree or disagree.
    If the comment is directly or indirectly agreeing with the target, or opposing or critizing something opposed to the target, then the opinion should be agree.
    If the comment is directly or indirectly disagreeing with the target, or opposing or critizing something in favor of the target, then the opinion should be disagree."""

    post = dspy.InputField(desc="""The post being commented on, may be useful in determining the relevancy or opinion of the comment""")
    parent_comment = dspy.InputField(desc="""The parent comment being replied to, may be useful in determining the relevancy or opinion of the comment.""")
    comment = dspy.InputField(desc="""The comment to determine the opinion of.""")
    target_opinion = dspy.InputField(desc="""The target that is either being agreed with, disagreed with, or not discussed.""")
    target_explanation = dspy.InputField(desc="""An explanation of the target, may be useful in determining the opinion of the comment.""")
    presence_of_opinion = dspy.OutputField(desc="""Choice between yes and no""")
    opinion = dspy.OutputField(desc="""Choice between agree and disagree.""")

class YesNoCommentStanceDetectionSignature(dspy.Signature):
    """Is the comment agreeing with the target opinion?
    If the comment is directly or indirectly agreeing with the opinion, or opposing or critizing something opposed to the opinion, then the answer is yes.
    If the comment is not agreeing with the opinion, or if the comment is not discussing the opinion, then the answer is no."""

    target_opinion = dspy.InputField(desc="""The target opinion.""")
    target_explanation = dspy.InputField(desc="""An explanation of the target opinion, may be useful in determining the opinion of the comment.""")
    post = dspy.InputField(desc="""The post being commented on, may be useful in determining the opinion of the comment""")
    parent_comment = dspy.InputField(desc="""The parent comment being replied to, may be useful in determining the opinion of the comment.""")
    comment = dspy.InputField(desc="""The comment to determine the opinion of.""")
    answer = dspy.OutputField(desc="""Answer yes or no""")

    def extract(example, raw_pred):
        field_names = ['target_opinion', 'target_explanation', 'post', 'parent_comment', 'comment', 'answer']
        return _extract_example(field_names, example, raw_pred)

class PostStanceDetectionSignature(dspy.Signature):
    """Determine the stance of the post towards the target."""

    target = dspy.InputField(desc="""The target of the stance detection.""")
    target_explanation = dspy.InputField(desc="""An explanation of the target, may be useful in determining the stance of the post.""")
    post = dspy.InputField(desc="""The post being commented on.""")
    parent_comment = dspy.InputField(desc="""The comment being replied to.""")
    comment = dspy.InputField(desc="""The comment to determine the stance of.""")
    stance = dspy.OutputField(desc="""Choice between favor, neutral, and against.""")



class ChainOfThoughtForOneStepOpinion(dspy.Predict):
    def __init__(self, signature, **config):
        super().__init__(signature, **config)

        signature = self.signature
        *keys, last_key = signature.kwargs.keys()

        rationale = dsp.Type(desc="""${predict the stance}""", 
                             prefix="Reasoning: Take a deep breath, and work on this problem step-by-step. Let's consider first if the comment is clearly expressing a stance on the target issue. ")
        self.extended_prompt = rationale.prefix + rationale.desc

        extended_kwargs = {key: signature.kwargs[key] for key in keys}
        extended_kwargs.update({'rationale': rationale, last_key: signature.kwargs[last_key]})
        
        self.extended_signature = dsp.Template(signature.instructions, **extended_kwargs)
    
    def forward(self, **kwargs):
        signature = self.extended_signature
        
        return super().forward(signature=signature, **kwargs)


class DSPyStanceDataset(DSPyDataset):
    def __init__(self, examples, target, target_explanation, train_num, val_num, strategy='order'):
        ds = []
        for ex in examples:
            d = ex.copy()
            d['target_opinion'] = target
            d['target_explanation'] = target_explanation
            ds.append(d)

        if strategy == 'order':
            self._train = ds[:train_num]
            self._dev = ds[train_num:train_num + val_num]
            self._test = ds[train_num + val_num:]
        elif 'ratio' in strategy:
            ratio = strategy.split(':')[1]
            num_favor, num_against, num_neutral = [int(x) for x in ratio]
            num_total = num_favor + num_against + num_neutral
            share_favor, share_against, share_neutral = num_favor / num_total, num_against / num_total, num_neutral / num_total
            favor = [d for d in ds if d['gold_stance'] == 'favor']
            against = [d for d in ds if d['gold_stance'] == 'against']
            neutral = [d for d in ds if d['gold_stance'] == 'neutral']
            self._train = favor[:int(share_favor * train_num)] + against[:int(share_against * train_num)] + neutral[:int(share_neutral * train_num)]
            self._dev = favor[int(share_favor * train_num):int(share_favor * train_num) + int(share_favor * val_num)] + against[int(share_against * train_num):int(share_against * train_num) + int(share_against * val_num)] + neutral[int(share_neutral * train_num):int(share_neutral * train_num) + int(share_neutral * val_num)]
            self._test = favor[int(share_favor * train_num) + int(share_favor * val_num):] + against[int(share_against * train_num) + int(share_against * val_num):] + neutral[int(share_neutral * train_num) + int(share_neutral * val_num):]
            random.Random(0).shuffle(self._train)
            random.Random(0).shuffle(self._dev)
            random.Random(0).shuffle(self._test)
        super().__init__(self, train_size=len(self._train), dev_size=len(self._dev), test_size=len(self._test))
        self.do_shuffle = False

class StanceDataset:
    def __init__(self, examples, target, train_num=None, val_num=None, backend='dspy', strategy='order'):
        self.backend = backend
        self.train_num = train_num
        self.val_num = val_num
        examples = [map_row(r) for r in examples]
        if self.backend == 'dspy':
            self.dataset = DSPyStanceDataset(examples, TARGET_NAMES[target], TARGET_EXPLANATIONS[target], train_num, val_num, strategy=strategy)
        elif self.backend == 'hf':
            self.dataset = examples

    def get_test_data(self):
        if self.backend == 'dspy':
            return [x.with_inputs('post', 'parent_comment', 'comment', 'target_opinion', 'target_explanation') for x in self.dataset.test]
        elif self.backend == 'hf':
            return self.dataset[self.train_num + self.val_num:]
        
    def get_train_data(self):
        if self.backend == 'dspy':
            return [x.with_inputs('post', 'parent_comment', 'comment', 'target_opinion', 'target_explanation') for x in self.dataset.train]
        elif self.backend == 'hf':
            return self.dataset[:self.train_num]

    def get_dev_data(self):
        if self.backend == 'dspy':
            return [x.with_inputs('post', 'parent_comment', 'comment', 'target_opinion', 'target_explanation') for x in self.dataset.dev]
        elif self.backend == 'hf':
            return self.dataset[self.train_num:self.train_num + self.val_num]
        

class StancesDataset(StanceDataset):
    def __init__(self, stance_datasets):
        self.backend = 'dspy'
        self.dataset = DSPyDataset(
            train_size=sum([d.dataset.train_size for d in stance_datasets]),
            dev_size=sum([d.dataset.dev_size for d in stance_datasets]),
            test_size=sum([d.dataset.test_size for d in stance_datasets])
        )
        self.dataset._train = []
        for d in stance_datasets:
            self.dataset._train += d.dataset._train
        self.dataset._dev = []
        for d in stance_datasets:
            self.dataset._dev += d.dataset._dev
        self.dataset._test = []
        for d in stance_datasets:
            self.dataset._test += d.dataset._test

        random.Random(0).shuffle(self.dataset._train)
        random.Random(0).shuffle(self.dataset._dev)
        random.Random(0).shuffle(self.dataset._test)


class StanceClassifier:
    def __init__(self, model_name, model_prompt_template=None, prompting_method='predict', opinion_method='twostep', backend='dspy', teleprompter='bootstrap'):
        self.backend = backend
        if self.backend == 'hf':
            if model_name == 'mistral':
                self.model = lm.Mistral()
            elif model_name == 'zephyr':
                self.model = lm.Zephyr()
            self.model_name = self.model.model_name

            for response in PROMPT_RESPONSES:
                if response['name'] == prompting_method:
                    self.regex = response['regex']
                    break
            else:
                raise ValueError(f"Invalid response type: {prompting_method}")

        elif self.backend == 'dspy':
            self.model_name = model_name
            self.model_prompt_template = model_prompt_template
            if "tune" not in teleprompter:
                if "gpt" in model_name:
                    self.model = dspy.OpenAI(self.model_name, os.environ['OPENAI_API_KEY'])
                else:
                    self.model = dspy.HFModel(self.model_name, model_prompt_template=self.model_prompt_template, model_kwargs={'torch_dtype': torch.bfloat16, 'use_flash_attention_2': True, 'device_map': 'auto'})
                    self.model.kwargs['pad_token_id'] = self.model.tokenizer.pad_token_id
                    self.model.kwargs['eos_token_id'] = self.model.tokenizer.eos_token_id
                dspy.settings.configure(lm=self.model)
        
        self.teleprompter = teleprompter
        self.teleprompter_settings = {}
        self.prompting_method = prompting_method
        self.opinion_method = opinion_method
        self.classifier = None
        self.shot_num = 0

    def _get_prompt_template(self, is_comment, is_replying):
        if self.backend == 'hf':
            return get_prompt_template(is_comment, is_replying)
        elif self.backend == 'dspy':
            if self.opinion_method == 'onestep':
                if is_comment:
                    return CommentStanceDetectionSignature({'demos':[]})
                else:
                    return PostStanceDetectionSignature({'demos':[]})
            elif self.opinion_method == 'twostep':
                TwoStepCommentStanceDetectionSignature({'demos': []})
            elif self.opinion_method == 'yesno':
                return YesNoCommentStanceDetectionSignature({'demos': []})
            elif self.opinion_method == 'template':
                return CommentStanceDetectionTemplateSignature({'demos': []})
            else:
                raise ValueError(f"Invalid opinion method: {self.opinion_method}")

    def _format_text_to_prompt(self, text, target):
        is_comment, is_replying = False, False
        if "comment" in text:
            is_comment = True
        if "parent_comment" in text:
            is_replying = True
        prompt = self._get_prompt_template(is_comment, is_replying)
        return prompt.format(**text, target=target)

    def _get_model_response(self, prompts, regex):
        response = self.model(prompts, regex=regex)
        return response

    def _predict_stance(self, texts, target):
        self._extra_responses = []
        if self.backend == 'hf':
            prompts = [self._format_text_to_prompt(text, target) for text in texts]
            responses = self._get_model_response(prompts, self.regex)

            def get_choice(r):
                if r.endswith('favor'):
                    return 'favor'
                elif r.endswith('against'):
                    return 'against'
                elif r.endswith('neutral'):
                    return 'neutral'

            responses = [get_choice(r) for r in responses]
        elif self.backend == 'dspy':
            responses = []
            for input in texts:
                classifier, config = self._get_classifier()

                if self.opinion_method != 'yesno':
                    if self.shot_num == 0:
                        self.classifier = classifier

                    response = self.classifier(config=config, **input.inputs()._store)

                    extras = {'opinion': response.opinion}
                    if 'rationale' in response.keys():
                        extras['rationale'] = response.rationale
                    self._extra_responses.append(extras)

                    stance = _parse_opinion_answer(response.opinion)
                elif self.opinion_method == 'yesno':
                    if self.shot_num == 0:
                        self.agree_classifier = classifier
                        self.disagree_classifier = classifier

                    inputs = input.inputs()._store
                    agree_inputs = inputs.copy()
                    agree_inputs['target_opinion'] = f"Support for {inputs['target_opinion']}"
                    agree_prediction = self.agree_classifier(config=config, **agree_inputs)
                    disagree_inputs = inputs.copy()
                    disagree_inputs['target_opinion'] = f"Against {inputs['target_opinion']}"
                    disagree_prediction = self.disagree_classifier(config=config, **disagree_inputs)

                    extras = {
                        'agree_answer': agree_prediction.answer,
                        'disagree_answer': disagree_prediction.answer,
                    }

                    if 'rationale' in agree_prediction.keys():
                        extras['agree_rationale'] = agree_prediction.rationale
                        extras['disagree_rationale'] = disagree_prediction.rationale

                    self._extra_responses.append(extras)
                    
                    agree_response = agree_prediction.answer
                    disagree_response = disagree_prediction.answer

                    agree_answer = _parse_yesno_answer(agree_response)
                    disagree_answer = _parse_yesno_answer(disagree_response)

                    assert agree_answer == 'yes' or agree_answer == 'no'
                    assert disagree_answer == 'yes' or disagree_answer == 'no'

                    if agree_answer == 'yes' and disagree_answer == 'no':
                        stance = 'favor'
                    elif agree_answer == 'no' and disagree_answer == 'yes':
                        stance = 'against'
                    elif agree_answer == 'no' and disagree_answer == 'no':
                        stance = 'neutral'
                    else:
                        stance = 'neutral'

                assert stance in ['favor', 'against', 'neutral']

                responses.append(stance)

        return responses

    def predict_stances(self, texts, target=None):
        if target is not None:
            self.target = target
        return self._predict_stance(texts, self.target)
    
    def set_target(self, target):
        self.target = target

    def train(self, trainset, valset=None, all_tasks=False, teleprompter_settings={}):

        self.shot_num = len(trainset)

        if self.opinion_method == 'yesno':
            # Validation logic: check that the predicted answer is correct.
            # Also check that the retrieved context does actually contain that answer.
            def validate_context_and_answer(example, pred, trace=None):
                pred_answer = _parse_yesno_answer(pred.answer)
                return example.answer == pred_answer
        else:
            # Validation logic: check that the predicted stance is correct.
            # Also check that the retrieved context does actually contain that stance.
            def validate_context_and_answer(example, pred, trace=None):
                pred_stance = _parse_opinion_answer(pred.opinion)
                return example.opinion == pred_stance

        # Compile!
        classifier, config = self._get_classifier()
        class StanceModule(dspy.Module):
            def __init__(self, task_map=None):
                self.classifier = classifier
                self.task_map = task_map
                super().__init__()
            
            def forward(self, **kwargs):
                kwargs.update(config)
                if self.task_map is not None:
                    kwargs['task_id'] = self.task_map[kwargs['target_opinion']]
                return self.classifier(**kwargs)

        if self.opinion_method == 'yesno':

            def convert_inputs(ex, new_set, agree):
                store = ex._store.copy()
                store['target_opinion'] = f"Support for {store['target_opinion']}" if agree else f"Against {store['target_opinion']}"
                stance = 'favor' if agree else 'against'
                store['answer'] = 'yes' if ex._store['gold_stance'] == stance else 'no'
                new_ex = dspy.Example(**store)
                new_set.append(new_ex.with_inputs('post', 'parent_comment', 'comment', 'target_opinion', 'target_explanation'))

            agree_trainset = []
            disagree_trainset = []
            for ex in trainset:
                convert_inputs(ex, agree_trainset, agree=True)
                convert_inputs(ex, disagree_trainset, agree=False)

            agree_valset = []
            disagree_valset = []
            for ex in valset:
                convert_inputs(ex, agree_valset, agree=True)
                convert_inputs(ex, disagree_valset, agree=False)

            # Set up a basic teleprompter, which will compile our RAG program.
            if len(valset) == 0:
                if self.teleprompter == 'bootstrap':
                    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)
                elif self.teleprompter == 'labelled':
                    teleprompter = LabeledFewShot(k=len(trainset))
                self.agree_classifier = teleprompter.compile(StanceModule(), trainset=agree_trainset)
                self.disagree_classifier = teleprompter.compile(StanceModule(), trainset=disagree_trainset)
            else:
                teleprompter = BootstrapFewShotWithOptuna(metric=validate_context_and_answer)
                self.agree_classifier = teleprompter.compile(StanceModule(), max_demos=len(agree_trainset), trainset=agree_trainset, valset=agree_valset)
                self.disagree_classifier = teleprompter.compile(StanceModule(), max_demos=len(disagree_trainset), trainset=disagree_trainset, valset=disagree_valset)

        else:
            for ex in trainset + valset:
                ex.opinion = ex.gold_stance

            if self.teleprompter == 'bootstrap':
                teleprompter = BootstrapFewShot(metric=validate_context_and_answer, max_labeled_demos=len(trainset), max_bootstrapped_demos=len(trainset))
                args = (StanceModule(),)
                kwargs = {'trainset': trainset}
            elif self.teleprompter == 'optuna':
                assert len(valset) > 0, "Optuna search requires a validation set"
                teleprompter = BootstrapFewShotWithOptuna(metric=validate_context_and_answer, max_labeled_demos=len(trainset), max_bootstrapped_demos=len(trainset))
                args = (StanceModule(),)
                kwargs = {'max_demos': len(trainset), 'trainset': trainset, 'valset': valset}
            elif self.teleprompter == 'random':
                assert len(valset) > 0, "Random search requires a validation set"
                teleprompter = BootstrapFewShotWithRandomSearch(metric=validate_context_and_answer, max_labeled_demos=len(trainset), max_bootstrapped_demos=len(trainset), num_threads=1)
                args = (StanceModule(),)
                kwargs = {'trainset': trainset, 'valset': valset}
            elif self.teleprompter == 'finetune':
                teleprompter = tuning.FineTune()
                args = (StanceModule(),)
                default_teleprompter_settings = {'method': 'ia3', 'lr': 1e-3, 'num_epochs': 50, 'gradient_accumulation_steps': 1}
                for k, v in default_teleprompter_settings.items():
                    if k not in teleprompter_settings:
                        teleprompter_settings[k] = v
                self.teleprompter_settings = teleprompter_settings
                kwargs = {'model_name': self.model_name, 'model_prompt_template': self.model_prompt_template, 'trainset': trainset, 'valset': valset, 'all_tasks': all_tasks}
                kwargs.update(self.teleprompter_settings)
            elif self.teleprompter == 'multitaskfinetune':
                teleprompter = tuning.FineTune()
                args = (StanceModule(),)
                default_teleprompter_settings = {'method': 'ia3', 'gradient_accumulation_steps': 1}
                default_teleprompter_settings['lr'] = 1e-3 if all_tasks else 5e-4
                default_teleprompter_settings['num_epochs'] = 50 if all_tasks else 10
                for k, v in default_teleprompter_settings.items():
                    if k not in teleprompter_settings:
                        teleprompter_settings[k] = v
                self.teleprompter_settings = teleprompter_settings
                kwargs = {'model_name': self.model_name, 'model_prompt_template': self.model_prompt_template, 'trainset': trainset, 'valset': valset, 'all_tasks': all_tasks}
                kwargs.update(self.teleprompter_settings)
            elif self.teleprompter == "prompttune":
                teleprompter = tuning.PromptTune()
                args = (StanceModule(),)
                self.teleprompter_settings = {'lr': 1e-3, 'num_epochs': 60, 'gradient_accumulation_steps': 8}
                kwargs = {'model_name': self.model_name, 'model_prompt_template': self.model_prompt_template, 'trainset': trainset, 'valset': valset, 'all_tasks': all_tasks}
                kwargs.update(self.teleprompter_settings)
            elif self.teleprompter == 'multitaskprompttune':
                teleprompter = tuning.MultiTaskPromptTune()
                args = (StanceModule(),)
                num_epochs = 50 if all_tasks else 10
                lr = 1e-4 if all_tasks else 1e-5
                self.teleprompter_settings = {'lr': lr, 'num_epochs': num_epochs, 'gradient_accumulation_steps': 8}
                kwargs = {'model_name': self.model_name, 'model_prompt_template': self.model_prompt_template, 'trainset': trainset, 'valset': valset, 'all_tasks': all_tasks}
                kwargs.update(self.teleprompter_settings)
                task_names = sorted(list(set([ex.target_opinion for ex in trainset + valset])))
                kwargs['task_map'] = {target: idx for idx, target in enumerate(task_names)}
            else:
                raise ValueError(f"Invalid teleprompter: {self.teleprompter}")
                
            self.classifier = teleprompter.compile(*args, **kwargs)
            self.checkpoint_path = teleprompter.checkpoint_path

    def _get_classifier(self, comment=True):
        if self.opinion_method == 'onestep':
            if comment:
                signature = CommentStanceDetectionSignature
            else:
                signature = PostStanceDetectionSignature
        elif self.opinion_method == 'twostep':
            signature = TwoStepCommentStanceDetectionSignature
        elif self.opinion_method == 'yesno':
            signature = YesNoCommentStanceDetectionSignature
        elif self.opinion_method == 'template':
            signature = CommentStanceDetectionTemplateSignature
        else:
            raise ValueError(f"Invalid opinion method: {self.opinion_method}")

        if self.prompting_method == 'predict':
            classifier = dspy.Predict(signature)
            if 'gpt' in self.model_name:
                config = {}
            else:
                config = {'max_tokens': 4}
        elif self.prompting_method == 'multicomparison':
            classifier = MultiComparison(signature)
            if 'gpt' in self.model_name:
                config = {}
            else:
                config = {'max_tokens': 4}
        elif self.prompting_method == 'chainofthought':
            classifier = dspy.ChainOfThought(signature)
            if 'gpt' in self.model_name:
                config = {}
            else:
                config = {'max_tokens': 400}
        elif self.prompting_method == 'chainofthoughtstance':
            classifier = ChainOfThoughtForOneStepOpinion(signature)
            if 'gpt' in self.model_name:
                config = {}
            else:
                config = {'max_tokens': 400}
        elif self.prompting_method == 'multichaincomparison':
            classifier = MultiChainComparison(signature)
            config = {}
        else:
            raise ValueError(f"Invalid prompting method: {self.prompting_method}")
        
        self.prompting_text = getattr(classifier, "extended_prompt", None)

        return classifier, config
    
    def get_extended_prompt(self):
        self._get_classifier()
        return self.prompting_text
    
    def remove_model(self):
        if getattr(self.classifier, 'lm', None) is not None:
            self.classifier.lm.model.to('cpu')
        else:
            self.classifier.predictors()[0].lm.model.to('cpu')

    def load_model(self, model_name, checkpoint_path, trainset):
        if self.teleprompter == 'finetune':
            teleprompter = tuning.FineTune()
        elif self.teleprompter == 'multitaskfinetune':
            teleprompter = tuning.FineTune()
        elif self.teleprompter == 'prompttune':
            teleprompter = tuning.PromptTune()
        elif self.teleprompter == 'multitaskprompttune':
            teleprompter = tuning.MultiTaskPromptTune()
        else:
            raise ValueError(f"Invalid teleprompter: {self.teleprompter}")

        classifier = self._get_classifier()[0]
        model_prompt_template = self.model_prompt_template
        self.classifier = teleprompter.load(model_name, checkpoint_path, trainset, classifier, model_prompt_template)
    
def _parse_yesno_answer(response):
    response = response.split('\n')[0].lower()
    if 'yes' in response and not 'no' in response:
        return 'yes'
    elif 'no' in response and not 'yes' in response:
        return 'no'
    else:
        if any(a in response.lower() for a in ['favor', 'agree', 'support']):
            return 'yes'
        elif any(a in response.lower() for a in ['against', 'disagree', 'unclear']):
            return 'no'
        else:
            return 'no'
        
def _parse_opinion_answer(opinion):
    opinion = opinion.split('\n')[0].lower()
    words = re.findall(r"(\w+)", opinion)
    def get_stance(word):
        if any(a in word for a in ['favor', 'agree', 'support']) and not 'disagree' in word:
            return 'favor'
        elif any(a in word for a in ['against', 'disagree']):
            return 'against'
        elif 'neutral' in word:
            return 'neutral'
        else:
            return None
    for word in words:
        stance = get_stance(word)
        if stance is not None:
            return stance
    else:
        return 'neutral'


class MultiChainComparison(dspy.Module):
    def __init__(self, signature, M=3, temperature=0.7, **config):
        super().__init__()

        self.M = M
        signature = dspy.Predict(signature).signature
        *keys, last_key = signature.kwargs.keys()

        extended_kwargs = {key: signature.kwargs[key] for key in keys}

        for idx in range(M):
            candidate_type = dsp.Type(prefix=f"Student Attempt #{idx+1}:", desc="${reasoning attempt}")
            extended_kwargs.update({f'reasoning_attempt_{idx+1}': candidate_type})
        
        rationale_type = dsp.Type(prefix="Accurate Reasoning: Thank you everyone. Let's now holistically", desc="${corrected reasoning}")
        extended_kwargs.update({'rationale': rationale_type, last_key: signature.kwargs[last_key]})

        signature = dsp.Template(signature.instructions, **extended_kwargs)
        self.predict = dspy.Predict(signature, temperature=temperature, **config)
        self.last_key = last_key

        self.chainofthought = dspy.ChainOfThought(signature, temperature=temperature, **config)
    
    def forward(self, **kwargs):
        attempts = []

        for _ in range(self.M):
            c = self.chainofthought(**kwargs)
            rationale = c.rationale.strip().split('\n')[0].strip()
            answer = _parse_opinion_answer(c[self.last_key])
            attempts.append(f"«{rationale} I'm not sure but my prediction is {answer}»")

        assert len(attempts) == self.M, len(attempts)

        kwargs = {**{f'reasoning_attempt_{idx+1}': attempt for idx, attempt in enumerate(attempts)}, **kwargs}
        return self.predict(**kwargs)
    
class MultiComparison(dspy.Module):
    def __init__(self, signature, M=3, temperature=0.4, **config):
        super().__init__()

        self.M = M
        self.predict = dspy.Predict(signature, temperature=temperature, **config)
    
    def forward(self, **kwargs):
        stance_counts = {}
        completions = []
        for _ in range(self.M):
            c = self.predict(**kwargs)
            completions.append(c)
            stance = _parse_opinion_answer(c.opinion)
            stance_counts[stance] = stance_counts.get(stance, 0) + 1

        stance_counts = sorted(stance_counts.items(), key=lambda x: x[1], reverse=True)
        stance = stance_counts[0][0]
        return [c for c in completions if _parse_opinion_answer(c.opinion) == stance][0]
    

def get_f1_score(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1

def get_fbeta_score(p, r, w):
    return (1 + w**2) * (p * r) / ((w**2 * p) + r) if p + r > 0 else 0

def get_stance_f1_score(gold_stances, stances, return_all=False, beta=0.5):

    num_f_tp = 0
    num_f_fp = 0
    num_f_fn = 0
    num_a_tp = 0
    num_a_fp = 0
    num_a_fn = 0
    num_n_tp = 0
    num_n_fp = 0
    num_n_fn = 0

    num_tf_pf = 0
    num_tf_pn = 0
    num_tf_pa = 0
    num_tn_pf = 0
    num_tn_pn = 0
    num_tn_pa = 0
    num_ta_pf = 0
    num_ta_pn = 0
    num_ta_pa = 0

    for gold_stance, stance in zip(gold_stances, stances):
        assert stance in ['favor', 'against', 'neutral']
        assert gold_stance in ['favor', 'against', 'neutral']
        if stance == 'favor' and gold_stance == 'favor':
            num_f_tp += 1
        if stance == 'favor' and gold_stance != 'favor':
            num_f_fp += 1
        if stance != 'favor' and gold_stance == 'favor':
            num_f_fn += 1
        if stance == 'against' and gold_stance == 'against':
            num_a_tp += 1
        if stance == 'against' and gold_stance != 'against':
            num_a_fp += 1
        if stance != 'against' and gold_stance == 'against':
            num_a_fn += 1
        if stance == 'neutral' and gold_stance == 'neutral':
            num_n_tp += 1
        if stance == 'neutral' and gold_stance != 'neutral':
            num_n_fp += 1
        if stance != 'neutral' and gold_stance == 'neutral':
            num_n_fn += 1

        if stance == 'favor' and gold_stance == 'favor':
            num_tf_pf += 1
        elif stance == 'neutral' and gold_stance == 'favor':
            num_tf_pn += 1
        elif stance == 'against' and gold_stance == 'favor':
            num_tf_pa += 1
        elif stance == 'favor' and gold_stance == 'neutral':
            num_tn_pf += 1
        elif stance == 'neutral' and gold_stance == 'neutral':
            num_tn_pn += 1
        elif stance == 'against' and gold_stance == 'neutral':
            num_tn_pa += 1
        elif stance == 'favor' and gold_stance == 'against':
            num_ta_pf += 1
        elif stance == 'neutral' and gold_stance == 'against':
            num_ta_pn += 1
        elif stance == 'against' and gold_stance == 'against':
            num_ta_pa += 1

    # calculate total F1 score as average of F1 scores for each stance
    # calculate f1 score for favor
    # calculate precision for favor

    favor_precision, favor_recall, favor_f1, favor_fbeta = 0, 0, 0, 0
    against_precision, against_recall, against_f1, against_fbeta = 0, 0, 0, 0
    neutral_precision, neutral_recall, neutral_f1, neutral_fbeta = 0, 0, 0, 0
    f1, precision, recall, fbeta = 0, 0, 0, 0

    if (num_f_tp + num_f_fn) > 0:
        favor_precision, favor_recall, favor_f1 = get_f1_score(num_f_tp, num_f_fp, num_f_fn)
        favor_fbeta = get_fbeta_score(favor_precision, favor_recall, beta)

    if (num_a_tp + num_a_fn) > 0:
        against_precision, against_recall, against_f1 = get_f1_score(num_a_tp, num_a_fp, num_a_fn)
        against_fbeta = get_fbeta_score(against_precision, against_recall, beta)

    if (num_n_tp + num_n_fn) > 0:
        neutral_precision, neutral_recall, neutral_f1 = get_f1_score(num_n_tp, num_n_fp, num_n_fn)
        neutral_fbeta = get_fbeta_score(neutral_precision, neutral_recall, beta)

    if (num_f_tp + num_f_fn) > 0 and (num_a_tp + num_a_fn) > 0:
        f1 = (favor_f1 + against_f1) / 2
        fbeta = (favor_fbeta + against_fbeta) / 2
        precision = (favor_precision + against_precision) / 2
        recall = (favor_recall + against_recall) / 2
    elif (num_f_tp + num_f_fn) > 0:
        f1 = favor_f1
        fbeta = favor_fbeta
        precision = favor_precision
        recall = favor_recall
    elif (num_a_tp + num_a_fn) > 0:
        f1 = against_f1
        fbeta = against_fbeta
        precision = against_precision
        recall = against_recall
    else:
        f1 = 0
        fbeta = 0
        precision = 0
        recall = 0

    if return_all:
        return {
            'favor': {
                'precision': favor_precision,
                'recall': favor_recall,
                'f1': favor_f1,
                f'f{beta}': favor_fbeta
            },
            'against': {
                'precision': against_precision,
                'recall': against_recall,
                'f1': against_f1,
                f'f{beta}': against_fbeta
            },
            'neutral': {
                'precision': neutral_precision,
                'recall': neutral_recall,
                'f1': neutral_f1,
                f'f{beta}': neutral_fbeta
            },
            'macro': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                f'f{beta}': fbeta
            },
            'test_num': len(gold_stances),
            'true_favor': {
                'predicted_favor': num_tf_pf,
                'predicted_neutral': num_tf_pn,
                'predicted_against': num_tf_pa
            },
            'true_neutral': {
                'predicted_favor': num_tn_pf,
                'predicted_neutral': num_tn_pn,
                'predicted_against': num_tn_pa
            },
            'true_against': {
                'predicted_favor': num_ta_pf,
                'predicted_neutral': num_ta_pn,
                'predicted_against': num_ta_pa
            }
        }
    else:
        return precision, recall, f1, fbeta