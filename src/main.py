from InputProcessor import InputProcessor


class TRag:
    """
    🦖🦖🦖
    Translation using LLM with Retrieval Augmented Generation (RAG)
    """

    def __init__(self,
                 input_processor: InputProcessor = None,
                 generation_model = None
                 ):
        if input_processor is None:
            self.input_processor = InputProcessor()

        if generation_model is None:
            self.generation_model = None
            raise ValueError()

    def __call__(self, *args, **kwargs):
        return self.translate(*args, **kwargs)

    def translate(self, text, input_processor_args={}, generation_args={}):
        prompt = self.input_processor.build_prompt(text, **input_processor_args)
        generation_output = self.generation_model(
            prompt=prompt,
            **generation_args
        )
        return generation_output
