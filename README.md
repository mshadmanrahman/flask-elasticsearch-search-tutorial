# Flask Elasticsearch Search Tutorial

A comprehensive search tutorial demonstrating advanced search capabilities using Flask and Elasticsearch. This project implements 5 different search modes including traditional BM25, dense vector search (kNN), sparse semantic search (ELSER), and hybrid approaches.

## 🚀 Features

- **5 Search Modes**:
  - **BM25**: Traditional full-text search with pagination
  - **kNN**: Dense vector search using sentence transformers
  - **Hybrid #1**: RRF combining BM25 + kNN
  - **ELSER**: Sparse semantic search using Elastic's learned sparse encoder
  - **Hybrid #2**: RRF combining BM25 + ELSER

- **Advanced Features**:
  - Dense vector embeddings (384-dimensional)
  - Sparse vector tokens for semantic understanding
  - Reciprocal Rank Fusion (RRF) for hybrid search
  - Minimum relevance score filtering
  - Responsive Bootstrap UI
  - Real-time search with aggregations

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Elasticsearch Cloud account
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/flask-elasticsearch-search-tutorial.git
   cd flask-elasticsearch-search-tutorial
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   Create a `.env` file with your Elasticsearch credentials:
   ```bash
   ELASTIC_API_KEY=your_elasticsearch_api_key
   ELSER_MODEL_ID=.elser_model_2_linux-x86_64
   ```

5. **Set up Elasticsearch**:
   - Deploy an Elasticsearch cluster on Elastic Cloud
   - Enable ML nodes for ELSER functionality
   - Deploy the ELSER model in Kibana

6. **Index your data**:
   ```bash
   python search.py reindex-dense-elser
   ```

7. **Run the application**:
   ```bash
   python app.py
   ```

8. **Open your browser**:
   Navigate to `http://localhost:5001`

## 📊 Search Modes Explained

### BM25 (Traditional Search)
- Uses Elasticsearch's built-in BM25 algorithm
- Best for exact keyword matches
- Includes pagination and aggregations
- Filters results by minimum relevance score

### kNN (Dense Vector Search)
- Uses sentence-transformers for semantic similarity
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Best for finding semantically similar content
- Shows all relevant results without pagination

### Hybrid #1 (BM25 + kNN RRF)
- Combines lexical and semantic search
- Uses Reciprocal Rank Fusion for result merging
- Balances precision and recall
- Optimal for most search scenarios

### ELSER (Sparse Semantic Search)
- Uses Elastic's Learned Sparse EncodeR v2
- Generates sparse vector tokens for semantic understanding
- Best for complex semantic queries
- Requires ML nodes in Elasticsearch

### Hybrid #2 (BM25 + ELSER RRF)
- Combines traditional and sparse semantic search
- Maximum coverage of different search approaches
- Best for comprehensive search results

## 🔧 Configuration

### Environment Variables
- `ELASTIC_API_KEY`: Your Elasticsearch Cloud API key
- `ELSER_MODEL_ID`: ELSER model ID (default: `.elser_model_2_linux-x86_64`)

### Search Parameters
- `MIN_SCORE`: Minimum relevance score threshold (default: 0.1)
- `EMBEDDING_DIMS`: Dense vector dimensions (default: 384)
- `PAGE_SIZE`: Results per page for BM25 (default: 5)

## 📁 Project Structure

```
├── app.py                 # Flask application
├── search.py             # Elasticsearch search logic
├── data.json             # Sample documents
├── requirements.txt      # Python dependencies
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   └── document.html
├── static/              # Static assets
│   └── elastic-logo.svg
└── .env                 # Environment variables (not in repo)
```

## 🎯 Usage Examples

### Basic Search
1. Enter a search query (e.g., "work from home")
2. Select a search mode from the dropdown
3. Click "Search" to see results

### Advanced Filtering
- Use `category:policy` to filter by category
- Use `year:2023` to filter by year
- Combine filters with search terms

### Comparing Search Modes
- Try the same query with different modes
- Notice how results and rankings differ
- ELSER often provides more semantically relevant results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Elasticsearch](https://www.elastic.co/) for the search engine
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Sentence Transformers](https://www.sbert.net/) for dense embeddings
- [Bootstrap](https://getbootstrap.com/) for the UI framework

## 📚 Additional Resources

- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [ELSER Model Guide](https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)