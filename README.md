# Story Recommendation System

This project implements a **Story Recommendation System** that uses **user interactions** and **story metadata** to suggest relevant stories to users.

## Datasets

- **[user_interaction.csv](https://drive.google.com/file/d/1qareCMhMjVYJN8wExn8AZnn9RC8DnUIe/view?usp=drive_link)**: Contains user-story interaction data.
  - Columns: `user_id`, `pratilipi_id`, `read_percentage`, `updated_at`.
  
- **[metadata.csv](https://drive.google.com/file/d/1QSkq4mcKY5-f9Xyt_nQPGNSSR46kBX5_/view?usp=sharing)**: Contains metadata about the stories.
  - Columns: `author_id`, `pratilipi_id`, `category_name`, `reading_time`, `updated_at`,`published_at`.

## Features

- **Content-Based Filtering**: Recommends stories based on metadata (e.g., category, author).
- **Collaborative Filtering**: Suggests stories based on user similarities or story similarities.
- **Hybrid Approach**: Combines both filtering methods.
- **Algorithms**: Includes **SVD**, **KNN**, **co-clustering**, and **cosine similarity**.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Hirenkhokhar/Story_recommendation_system.git
   ```

2. Install dependencies:
   ```bash
   cd Story_recommendation_system
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

3. Run the system:
   ```bash
   python main.py
   ```

### **Links:**
- **user_interaction.csv**: [user_interaction.csv](https://drive.google.com/file/d/1qareCMhMjVYJN8wExn8AZnn9RC8DnUIe/view?usp=drive_link)
- **metadata.csv**: [metadata.csv](https://drive.google.com/file/d/1QSkq4mcKY5-f9Xyt_nQPGNSSR46kBX5_/view?usp=sharing)

---
