import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import json

class InferlessPythonModel:
    def initialize(self):
        self.module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3')

    def infer(self, inputs):
        signature_name = inputs['signature_name']
        instances = inputs['instances']
        instances = json.loads(instances)
        
        embeddings = []
        if signature_name == 'question_encoder':
            embeddings = self.module.signatures['question_encoder'](tf.constant(instances))
        else:
            responses = []
            response_contexts = []
            # instances = '[{"input": "hello", "context": "how are you"}]'
            
            for instance in instances:
                responses = instance['input']
                response_contexts = instance['context']

            embeddings = self.module.signatures['response_encoder'](input=tf.constant(responses), context=tf.constant(response_contexts))

        print("The shape is " + str(embeddings["outputs"].numpy().shape))
        return {"predictions" : embeddings["outputs"].numpy() }

    def finalize(self):
        self.module = None

