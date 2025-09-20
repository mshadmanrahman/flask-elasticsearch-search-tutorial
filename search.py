import json
from pprint import pprint
import os
import time

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()


class Search:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.es = Elasticsearch(
            hosts=['https://3e5cd53f1eec4916ad122c933fa350c2.us-central1.gcp.cloud.es.io:443'],
            api_key=os.environ['ELASTIC_API_KEY']
        )
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def create_index(self):
        self.es.indices.delete(index='my_documents', ignore_unavailable=True)
        self.es.indices.create(
            index='my_documents',
            mappings={
                'properties': {
                    'embedding': {
                        'type': 'dense_vector',
                    },
                    'elser_embedding': {
                        'type': 'sparse_vector',
                    },
                }
            },
            settings={
                'index': {
                    'default_pipeline': 'elser-ingest-pipeline'
                }
            }
        )

    def get_embedding(self, text):
        return self.model.encode(text)

    def insert_document(self, document):
        return self.es.index(index='my_documents', document={
            **document,
            'embedding': self.get_embedding(document['summary']),
        })

    def insert_documents(self, documents):
        operations = []
        for document in documents:
            operations.append({'index': {'_index': 'my_documents'}})
            operations.append({
                **document,
                'embedding': self.get_embedding(document['summary']),
            })
        return self.es.bulk(operations=operations)

    def reindex(self):
        self.create_index()
        with open('data.json', 'rt') as f:
            documents = json.loads(f.read())
        return self.insert_documents(documents)

    def deploy_elser(self):
        # download ELSER v2
        self.es.ml.put_trained_model(model_id='.elser_model_2',
                                     input={'field_names': ['text_field']})
        
        # wait until ready
        while True:
            status = self.es.ml.get_trained_models(model_id='.elser_model_2',
                                                   include='definition_status')
            if status['trained_model_configs'][0]['fully_defined']:
                # model is ready
                break
            time.sleep(1)

        # deploy the model
        self.es.ml.start_trained_model_deployment(model_id='.elser_model_2')

        # define a pipeline
        self.es.ingest.put_pipeline(
            id='elser-ingest-pipeline',
            processors=[
                {
                    'inference': {
                        'model_id': '.elser_model_2',
                        'input_output': [
                            {
                                'input_field': 'summary',
                                'output_field': 'elser_embedding',
                            }
                        ]
                    }
                }
            ]
        )

    def search(self, **query_args):
        # sub_searches is not currently supported in the client, so we send
        # search requests using the body argument
        if 'from_' in query_args:
            query_args['from'] = query_args['from_']
            del query_args['from_']
        
        # Check if this is a sub_searches query (for hybrid ELSER)
        if 'sub_searches' in query_args:
            # For now, fall back to ELSER search only since sub_searches is not supported
            # This is a simplified implementation
            try:
                elser_query = {
                    'query': query_args['sub_searches'][1]['query'],  # Use the ELSER query
                    'size': query_args.get('size', 5),
                    'from': query_args.get('from', 0)
                }
                if 'aggs' in query_args:
                    elser_query['aggs'] = query_args['aggs']
                return self.es.search(
                    index='my_documents',
                    **elser_query
                )
            except (KeyError, IndexError) as e:
                # If there's an error with the sub_searches structure, fall back to a simple ELSER query
                print(f"Warning: sub_searches structure error: {e}. Falling back to simple ELSER search.")
                return self.es.search(
                    index='my_documents',
                    query={
                        'text_expansion': {
                            'elser_embedding': {
                                'model_id': '.elser_model_2',
                                'model_text': 'work'  # Default search term
                            }
                        }
                    },
                    size=query_args.get('size', 5),
                    from_=query_args.get('from', 0)
                )
        else:
            return self.es.search(
                index='my_documents',
                **query_args
            )

    def retrieve_document(self, id):
        return self.es.get(index='my_documents', id=id)