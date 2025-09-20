# Flask Elasticsearch Search Tutorial

A comprehensive search tutorial demonstrating advanced search capabilities using Flask and Elasticsearch. This project implements 5 different search modes including traditional BM25, dense vector search (kNN), sparse semantic search (ELSER), and hybrid approaches with full Elastic Cloud integration.

## 🚀 Features

- **5 Search Modes**:
  - **BM25**: Traditional full-text search with multi-field matching
  - **kNN**: Dense vector search using sentence transformers
  - **Hybrid**: RRF combining BM25 + kNN with advanced ranking
  - **ELSER**: Sparse semantic search using Elastic's learned sparse encoder
  - **Hybrid ELSER**: ELSER search with fallback implementation

- **Advanced Features**:
  - **Elastic Cloud Integration** - Full support for Elastic Cloud deployments
  - **Faceted Search** - Category and year filtering with aggregations
  - **Pagination** - Complete pagination support for all search modes
  - **Dense Vector Embeddings** - 384-dimensional semantic vectors
  - **Sparse Vector Tokens** - ELSER-generated semantic understanding
  - **Reciprocal Rank Fusion (RRF)** - Advanced hybrid search ranking
  - **Error Handling** - Comprehensive fallbacks and graceful degradation
  - **Responsive UI** - Modern Bootstrap interface with search mode selection
  - **Real-time Search** - Live search with aggregations and filters

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- Elastic Cloud account (free trial available)
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mshadmanrahman/flask-elasticsearch-search-tutorial.git
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

4. **Configure Elastic Cloud**:
   Create a `.env` file with your Elastic Cloud credentials:
   ```bash
   ELASTIC_CLOUD_ID="your_cloud_id_here"
   ELASTIC_API_KEY="your_api_key_here"
   ```

5. **Deploy ELSER Model**:
   ```bash
   flask deploy-elser
   ```

6. **Index your data**:
   ```bash
   flask reindex
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
- Multi-field search across name, summary, and content
- Best for exact keyword matches and phrase queries
- Includes pagination, aggregations, and faceted search
- Supports category and year filtering

### kNN (Dense Vector Search)
- Uses sentence-transformers for semantic similarity
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Best for finding semantically similar content
- Understands meaning beyond exact word matches
- Excellent for conceptual queries

### Hybrid (BM25 + kNN RRF)
- Combines lexical and semantic search approaches
- Uses Reciprocal Rank Fusion for intelligent result merging
- Balances precision and recall for optimal results
- Best for most general search scenarios
- Provides comprehensive coverage

### ELSER (Sparse Semantic Search)
- Uses Elastic's Learned Sparse EncodeR v2 model
- Generates sparse vector tokens for semantic understanding
- Best for complex semantic and conceptual queries
- Automatically deployed and managed by Elasticsearch
- Requires Elastic Cloud or ML-enabled cluster

### Hybrid ELSER (ELSER with Fallback)
- ELSER search with intelligent fallback mechanisms
- Handles sub_searches limitations gracefully
- Provides robust semantic search capabilities
- Ensures reliable search experience

## 🔧 Configuration

### Environment Variables
- `ELASTIC_CLOUD_ID`: Your Elastic Cloud deployment ID
- `ELASTIC_API_KEY`: Your Elastic Cloud API key

### Search Parameters
- **Page Size**: 5 results per page (configurable)
- **Vector Dimensions**: 384 (sentence-transformers)
- **Model ID**: `.elser_model_2` (ELSER v2)
- **Minimum Score**: Dynamic based on search mode

## 📁 Project Structure

```
├── app.py                 # Flask application with all routes
├── search.py             # Core Elasticsearch search logic
├── data.json             # Sample documents for indexing
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (not in repo)
├── .gitignore           # Git ignore rules
├── templates/            # HTML templates
│   ├── base.html        # Base template with Bootstrap
│   ├── index.html       # Main search interface
│   └── document.html    # Document detail view
└── static/              # Static assets
    └── elastic-logo.svg # Elasticsearch logo
```

## 🎯 Usage Examples

### Basic Search
1. Enter a search query (e.g., "work from home", "team collaboration")
2. Select a search mode from the dropdown
3. Click "Search" to see results
4. Use pagination to browse through results

### Advanced Filtering
- Use `category:sharepoint` to filter by category
- Use `year:2023` to filter by year
- Combine filters with search terms: `category:teams work from home`
- View faceted search options in the sidebar

### Comparing Search Modes
- Try the same query with different modes to see variations
- Notice how BM25 finds exact matches while kNN finds semantic matches
- ELSER often provides more contextually relevant results
- Hybrid modes combine the best of both approaches

### Example Queries to Try
- **"remote work"** - Compare BM25 vs ELSER results
- **"employee benefits"** - See semantic understanding differences
- **"team collaboration"** - Test conceptual search capabilities
- **"HR policies"** - Explore different search approaches

## 🔍 Search Mode Differences

Different search modes excel at different types of queries:

| Query Type | BM25 | kNN | ELSER | Hybrid |
|------------|------|-----|-------|--------|
| Exact keywords | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Synonyms | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Conceptual | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Phrase matching | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Semantic similarity | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## 🚨 Troubleshooting

### Common Issues
1. **Connection Errors**: Verify your Elastic Cloud credentials in `.env`
2. **ELSER Not Working**: Ensure you've run `flask deploy-elser`
3. **No Results**: Try reindexing with `flask reindex`
4. **License Errors**: Some features require Elastic Cloud trial or paid plan

### Getting Help
- Check the Flask application logs for detailed error messages
- Verify your Elastic Cloud deployment is running
- Ensure all dependencies are installed correctly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Elasticsearch](https://www.elastic.co/) for the powerful search engine
- [Flask](https://flask.palletsprojects.com/) for the lightweight web framework
- [Sentence Transformers](https://www.sbert.net/) for dense embeddings
- [Bootstrap](https://getbootstrap.com/) for the responsive UI framework
- [Elastic Cloud](https://cloud.elastic.co/) for managed Elasticsearch

## 📚 Additional Resources

- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [ELSER Model Guide](https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Elastic Cloud Setup](https://www.elastic.co/guide/en/cloud/current/ec-getting-started.html)

## 🎉 Recent Updates

- ✅ **Complete Elastic Cloud Integration** - Full support for managed Elasticsearch
- ✅ **Enhanced Error Handling** - Robust fallbacks and graceful degradation
- ✅ **Improved UI** - Search mode selection and better user experience
- ✅ **Faceted Search** - Category and year filtering with aggregations
- ✅ **Pagination Support** - Complete pagination for all search modes
- ✅ **Security Improvements** - Proper credential management and .gitignore
- ✅ **Comprehensive Testing** - All search modes fully functional