from typing import Optional, List, Dict, Any, Union, Callable, AsyncGenerator
import logging
import torch
from functools import lru_cache
from packaging import version
import json

from transformers import PreTrainedTokenizer

import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from maga_transformer.openai.renderers.custom_renderer import CustomChatRenderer, \
    RendererParams, ProcessedOutput, StreamResponseObject
from maga_transformer.models.base_model import BaseTokenizer, GenerateOutput
from maga_transformer.openai.api_datatype import ChatMessage, GPTFunctionDefinition, RoleEnum, \
    ChatCompletionRequest, ChatCompletionResponseStreamChoice, DeltaMessage, FinisheReason, UsageInfo


DEFAULT_CHAT_API_TEMPLATE = (
    "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

# This class is designed to replace `PreTrainedTokenizer.apply_chat_template` functionality,
# providing more capability to customize the template.
# More specifically, this method allows template to use `functions` field, following openai chat api format.
# Besides that, other template elements is compatible with `PreTrainedTokenizer.apply_chat_template`.
class BasicRenderer(CustomChatRenderer):
    def __init__(self,
                 tokenizer: Union[PreTrainedTokenizer, BaseTokenizer],
                 renderer_params: RendererParams,
    ):
        super().__init__(tokenizer, renderer_params)

        if version.parse(jinja2.__version__) <= version.parse("3.0.0"):
            raise ImportError(
                "apply_chat_template requires jinja2>=3.0.0 to be installed. "
                "Your version is " f"{jinja2.__version__}."
            )

        self.add_generation_prompt = True
        self.chat_template = None
        self.special_tokens_map = {}

        try:
            self.chat_template = tokenizer.chat_template
            assert (self.chat_template != None)
        except:
            try:
                self.chat_template = tokenizer.default_chat_template
                assert (self.chat_template != None)
            except:
                logging.info(f"tokenizer {tokenizer} has no chat_template nor "
                                "default_chat_template attribute. Use default template.")
                self.chat_template = DEFAULT_CHAT_API_TEMPLATE
                self.extra_stop_word_ids_list.append(self.tokenizer.encode("<|im_end|>"))

        self.extra_stop_words = None
        try:
            self.extra_stop_words = tokenizer.additional_special_tokens
        except:
            pass
        if self.extra_stop_words == None or self.extra_stop_words == []:
            self.extra_stop_words = ["<|im_start|>", "<|im_end|>"]
        self.extra_stop_word_ids_list = [
            tokenizer.encode(stop_word) for stop_word in self.extra_stop_words
        ]
        self.stop_words_list.extend(self.extra_stop_words)
        self.stop_word_ids_list.extend(self.extra_stop_word_ids_list)

        logging.info(f"use chat template: [ {self.chat_template} ]  ")
        self.compiled_template = self._compile_jinja_template(self.chat_template)

    @lru_cache
    def _compile_jinja_template(self, chat_template) -> jinja2.Template:

        def raise_exception(message):
            raise TemplateError(message)

        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        jinja_env.globals["raise_exception"] = raise_exception
        return jinja_env.from_string(chat_template)

    def render_chat(self, request: ChatCompletionRequest) -> List[int]:
        rendered = self.compiled_template.render(
            messages=request.messages,
            functions=request.functions,
            json=json,
            add_generation_prompt=self.add_generation_prompt,
            **self.special_tokens_map
        )
        logging.debug(f"request [{request.model_dump_json(indent=4)}] rendered string: [{rendered}]]")
        return self.tokenizer.encode(rendered)

    async def render_response_stream(
            self,
            output_generator: AsyncGenerator[GenerateOutput, None],
            request: ChatCompletionRequest,
            input_token_length: int,
    ) -> AsyncGenerator[StreamResponseObject, None]:
        # TODO(wangyin): maybe deal with the case of multiple returns.
        index = 0
        yield StreamResponseObject(
            choices=[ChatCompletionResponseStreamChoice(
                index=index,
                delta=DeltaMessage(
                    role=RoleEnum.assistant,
                ),
            )]
        )

        responded_token_length = 0
        output_token_length = 0
        responded_string = ""
        finish_reason = None
        # TODO(wangyin): decode incrementally.
        async for output in output_generator:
            index += 1
            processed_output = self._process_output_ids_tensor(
                input_token_length + responded_token_length, output.output_ids)
            delta_output_string = processed_output.output_str
            delta_output_token_length = processed_output.output_token_length
            finish_reason = processed_output.finish_reason
            output_token_length = responded_token_length + delta_output_token_length
            if len(delta_output_string) > 0:
                responded_token_length += delta_output_token_length
                responded_string += delta_output_string
                yield StreamResponseObject(
                    choices=[ChatCompletionResponseStreamChoice(
                        index=index,
                        delta=DeltaMessage(
                            content=delta_output_string,
                        ),
                    )],
                    usage=UsageInfo(
                        prompt_tokens=input_token_length,
                        total_tokens=input_token_length + output_token_length,
                        completion_tokens=output_token_length
                    )
                )

        if finish_reason == None:
            logging.debug(f"output [{responded_string}] found no stop reason! use stop as default.")
            finish_reason = FinisheReason.stop

        yield StreamResponseObject(
            choices=[ChatCompletionResponseStreamChoice(
                index=index + 1,
                delta=DeltaMessage(
                    content="",
                ),
                finish_reason=finish_reason
            )],
            usage=UsageInfo(
                prompt_tokens=input_token_length,
                total_tokens=input_token_length + output_token_length,
                completion_tokens=output_token_length
            )
        )
