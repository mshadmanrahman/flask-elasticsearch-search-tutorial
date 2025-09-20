import re
from flask import Flask, render_template, request
from search import Search

app = Flask(__name__)
es = Search()


def extract_filters(query):
    filters = []

    filter_regex = r'category:([^\s]+)\s*'
    m = re.search(filter_regex, query)
    if m:
        filters.append({
            'term': {
                'category.keyword': {
                    'value': m.group(1)
                }
            },
        })
        query = re.sub(filter_regex, '', query).strip()

    filter_regex = r'year:([^\s]+)\s*'
    m = re.search(filter_regex, query)
    if m:
        filters.append({
            'range': {
                'updated_at': {
                    'gte': f'{m.group(1)}||/y',
                    'lte': f'{m.group(1)}||/y',
                }
            },
        })
        query = re.sub(filter_regex, '', query).strip()

    return {'filter': filters}, query


@app.get('/')
def index():
    return render_template('index.html')


@app.post('/')
def handle_search():
    query = request.form.get('query', '')
    mode = request.form.get('mode', 'bm25')
    filters, parsed_query = extract_filters(query)
    from_ = request.form.get('from_', type=int, default=0)

    if mode == 'bm25':
        if parsed_query:
            search_query = {
                'must': {
                    'multi_match': {
                        'query': parsed_query,
                        'fields': ['name', 'summary', 'content'],
                    }
                }
            }
        else:
            search_query = {
                'must': {
                    'match_all': {}
                }
            }

        results = es.search(
            query={
                'bool': {
                    **search_query,
                    **filters
                }
            },
            aggs={
                'category-agg': {
                    'terms': {
                        'field': 'category.keyword',
                    }
                },
                'year-agg': {
                    'date_histogram': {
                        'field': 'updated_at',
                        'calendar_interval': 'year',
                        'format': 'yyyy',
                    },
                },
            },
            size=5,
            from_=from_
        )
    elif mode == 'knn':
        results = es.search(
            knn={
                'field': 'embedding',
                'query_vector': es.get_embedding(parsed_query),
                'num_candidates': 50,
                'k': 10,
                **filters,
            },
            aggs={
                'category-agg': {
                    'terms': {
                        'field': 'category.keyword',
                    }
                },
                'year-agg': {
                    'date_histogram': {
                        'field': 'updated_at',
                        'calendar_interval': 'year',
                        'format': 'yyyy',
                    },
                },
            },
            size=5,
            from_=from_
        )
    elif mode == 'hybrid':
        if parsed_query:
            search_query = {
                'must': {
                    'multi_match': {
                        'query': parsed_query,
                        'fields': ['name', 'summary', 'content'],
                    }
                }
            }
        else:
            search_query = {
                'must': {
                    'match_all': {}
                }
            }

        results = es.search(
            query={
                'bool': {
                    **search_query,
                    **filters
                }
            },
            knn={
                'field': 'embedding',
                'query_vector': es.get_embedding(parsed_query),
                'k': 10,
                'num_candidates': 50,
                **filters,
            },
            rank={
                'rrf': {}
            },
            aggs={
                'category-agg': {
                    'terms': {
                        'field': 'category.keyword',
                    }
                },
                'year-agg': {
                    'date_histogram': {
                        'field': 'updated_at',
                        'calendar_interval': 'year',
                        'format': 'yyyy',
                    },
                },
            },
            size=5,
            from_=from_,
        )
    elif mode == 'elser':
        results = es.search(
            query={
                'bool': {
                    'must': [
                        {
                            'text_expansion': {
                                'elser_embedding': {
                                    'model_id': '.elser_model_2',
                                    'model_text': parsed_query,
                                }
                            },
                        }
                    ],
                    **filters,
                }
            },
            aggs={
                'category-agg': {
                    'terms': {
                        'field': 'category.keyword',
                    }
                },
                'year-agg': {
                    'date_histogram': {
                        'field': 'updated_at',
                        'calendar_interval': 'year',
                        'format': 'yyyy',
                    },
                },
            },
            size=5,
            from_=from_,
        )
    elif mode == 'hybrid_elser':
        if parsed_query:
            search_query = {
                'sub_searches': [
                    {
                        'query': {
                            'bool': {
                                'must': {
                                    'multi_match': {
                                        'query': parsed_query,
                                        'fields': ['name', 'summary', 'content'],
                                    }
                                },
                                **filters
                            }
                        }
                    },
                    {
                        'query': {
                            'bool': {
                                'must': [
                                    {
                                        'text_expansion': {
                                            'elser_embedding': {
                                                'model_id': '.elser_model_2',
                                                'model_text': parsed_query,
                                            }
                                        },
                                    }
                                ],
                                **filters,
                            }
                        }
                    },
                ],
                'rank': {
                    'rrf': {}
                },
            }
        else:
            search_query = {
                'query': {
                    'bool': {
                        'must': {'match_all': {}},
                        **filters
                    }
                }
            }

        results = es.search(
            **search_query,
            aggs={
                'category-agg': {
                    'terms': {
                        'field': 'category.keyword',
                    }
                },
                'year-agg': {
                    'date_histogram': {
                        'field': 'updated_at',
                        'calendar_interval': 'year',
                        'format': 'yyyy',
                    },
                },
            },
            size=5,
            from_=from_,
        )
    else:
        # Default to BM25
        if parsed_query:
            search_query = {
                'must': {
                    'multi_match': {
                        'query': parsed_query,
                        'fields': ['name', 'summary', 'content'],
                    }
                }
            }
        else:
            search_query = {
                'must': {
                    'match_all': {}
                }
            }

        results = es.search(
            query={
                'bool': {
                    **search_query,
                    **filters
                }
            },
            aggs={
                'category-agg': {
                    'terms': {
                        'field': 'category.keyword',
                    }
                },
                'year-agg': {
                    'date_histogram': {
                        'field': 'updated_at',
                        'calendar_interval': 'year',
                        'format': 'yyyy',
                    },
                },
            },
            size=5,
            from_=from_
        )

    aggs = {
        'Category': {
            bucket['key']: bucket['doc_count']
            for bucket in results['aggregations']['category-agg']['buckets']
        },
        'Year': {
            bucket['key_as_string']: bucket['doc_count']
            for bucket in results['aggregations']['year-agg']['buckets']
            if bucket['doc_count'] > 0
        },
    }
    return render_template('index.html', results=results['hits']['hits'],
                           query=query, from_=from_,
                           total=results['hits']['total']['value'], aggs=aggs, mode=mode)


@app.get('/document/<id>')
def get_document(id):
    document = es.retrieve_document(id)
    title = document['_source']['name']
    paragraphs = document['_source']['content'].split('\n')
    return render_template('document.html', title=title, paragraphs=paragraphs)


@app.cli.command()
def reindex():
    """Regenerate the Elasticsearch index."""
    response = es.reindex()
    print(f'Index with {len(response["items"])} documents created '
          f'in {response["took"]} milliseconds.')


@app.cli.command()
def deploy_elser():
    """Deploy the ELSER v2 model to Elasticsearch."""
    try:
        es.deploy_elser()
    except Exception as exc:
        print(f'Error: {exc}')
    else:
        print(f'ELSER model deployed.')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)