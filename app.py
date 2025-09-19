import re
from flask import Flask, render_template, request
from search import Search

app = Flask(__name__)
es = Search()

def extract_filters(query):
    filters = []
    m = re.search(r'category:([^\s]+)\s*', query)
    if m:
        filters.append({"term": {"category.keyword": {"value": m.group(1)}}})
        query = re.sub(r'category:[^\s]+\s*', '', query).strip()
    m = re.search(r'year:([^\s]+)\s*', query)
    if m:
        filters.append({"range": {"updated_at": {"gte": f"{m.group(1)}||/y", "lte": f"{m.group(1)}||/y"}}})
        query = re.sub(r'year:[^\s]+\s*', '', query).strip()
    return {"filter": filters}, query

@app.route('/', methods=['GET', 'POST'])
def handle_search():
    query = request.form.get('query', '')
    mode = request.form.get('mode', 'bm25')
    filters, parsed_query = extract_filters(query)
    from_ = int(request.form.get('from_', 0))
    page_size = 5 if mode == 'bm25' else 50

    if mode == 'bm25':
        results = es.search_bm25(query=parsed_query, filters=filters, size=page_size, from_=from_)
    elif mode == 'knn':
        results = es.search_knn(query=parsed_query or query, filters=filters, size=page_size, from_=0)
    elif mode == 'hybrid':
        results = es.search_hybrid_rrf(query=parsed_query or query, filters=filters, size=page_size, from_=0)
    elif mode == 'elser':
        results = es.search_elser(query=parsed_query or query, filters=filters, size=page_size, from_=0)
    elif mode == 'hybrid_lexical_elser':
        results = es.search_hybrid_rrf_lexical_elser(query=parsed_query or query, filters=filters, size=page_size)
    else:
        results = es.search_bm25(query=parsed_query, filters=filters, size=page_size, from_=from_)
    aggs = {}
    if 'aggregations' in results:
        aggs = {
            "Category": {b['key']: b['doc_count'] for b in results['aggregations']['category-agg']['buckets']},
            "Year": {b['key_as_string']: b['doc_count'] for b in results['aggregations']['year-agg']['buckets'] if b['doc_count'] > 0}
        }
    return render_template('index.html', results=results['hits']['hits'], query=query,
                           from_=from_, total=results['hits']['total']['value'], aggs=aggs, mode=mode)

@app.route('/document/<id>')
def get_document(id):
    document = es.retrieve_document(id)
    title = document['_source']['name']
    paragraphs = document['_source']['content'].split('\n')
    return render_template('document.html', title=title, paragraphs=paragraphs)

if __name__ == "__main__":
    app.run(port=5001)
