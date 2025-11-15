# Create project folder
mkdir -p /var/www/embedding_service
cd /var/www/embedding_service

# Create virtual environment
python3 -m venv /var/www/embeddings_env
source /var/www/embeddings_env/bin/activate

# Install required packages
pip install sentence-transformers torch numpy


[Unit]
Description=Local Embedding Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/embedding_service
Environment="PATH=/var/www/embeddings_env/bin"
ExecStart=/var/www/embeddings_env/bin/python3 /var/www/embedding_service/embedding_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target