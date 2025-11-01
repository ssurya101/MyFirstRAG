import streamlit as st
import pandas as pd
from pymongo import MongoClient
from openai import OpenAI
import os
from datetime import datetime

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_key")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# Initialize MongoDB connection
@st.cache_resource
def get_mongo_client(connection_string):
    try:
        client = MongoClient(connection_string)
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"MongoDB connection failed: {str(e)}")
        return None

# Generate embeddings
def get_embedding(text, client):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Vector search in MongoDB
def vector_search(query_embedding, collection, limit=3):
    try:
        results = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "issue": 1,
                    "solution": 1,
                    "category": 1,
                    "timestamp": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ])
        return list(results)
    except Exception as e:
        st.error(f"Vector search error: {str(e)}")
        return []

# Generate solution using LLM
def generate_solution(user_issue, similar_issues, client):
    context = "\n\n".join([
        f"Past Issue: {issue['issue']}\nSolution: {issue['solution']}\nCategory: {issue['category']}"
        for issue in similar_issues
    ])
    
    prompt = f"""You are an expert technical support assistant. Based on similar past issues and their solutions, provide a clear solution for the new issue.

Past Issues and Solutions:
{context}

New Issue:
{user_issue}

Provide a detailed solution based on the similar past issues. If the new issue is very similar to a past issue, adapt that solution. If it's different, use your expertise to provide a helpful solution."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful technical support assistant specializing in issue remediation."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Load data from CSV and insert into MongoDB
def load_csv_to_mongodb(csv_file, collection, openai_client):
    df = pd.read_csv(csv_file)
    
    required_columns = ['issue', 'solution', 'category']
    if not all(col in df.columns for col in required_columns):
        st.error(f"CSV must contain columns: {', '.join(required_columns)}")
        return False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        issue_text = f"{row['issue']} {row['category']}"
        embedding = get_embedding(issue_text, openai_client)
        
        doc = {
            "issue": row['issue'],
            "solution": row['solution'],
            "category": row['category'],
            "embedding": embedding,
            "timestamp": datetime.now()
        }
        
        collection.insert_one(doc)
        progress_bar.progress((idx + 1) / len(df))
        status_text.text(f"Loaded {idx + 1}/{len(df)} records")
    
    progress_bar.empty()
    status_text.empty()
    return True

# Add new issue to knowledge base
def add_issue_to_kb(issue, solution, category, collection, openai_client):
    issue_text = f"{issue} {category}"
    embedding = get_embedding(issue_text, openai_client)
    
    doc = {
        "issue": issue,
        "solution": solution,
        "category": category,
        "embedding": embedding,
        "timestamp": datetime.now()
    }
    
    collection.insert_one(doc)

# Streamlit UI
def main():
    st.set_page_config(page_title="Issue Remediation System", page_icon="🔧", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            background-color: #fafafa;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #e3f2fd;
            border-radius: 8px 8px 0 0;
            padding: 12px 24px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #2196f3;
            color: white;
        }
        h1 {
            color: #1565c0;
        }
        h2, h3 {
            color: #0d47a1;
        }
        .stButton>button {
            background-color: #2196f3;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 12px 24px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1976d2;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔧 Automated Issue Remediation System")
    st.markdown("*RAG-powered technical support using MongoDB Vector Search & OpenAI*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
        mongo_uri = st.text_input("MongoDB Connection String", type="password", 
                                   placeholder="mongodb+srv://username:password@cluster.mongodb.net/")
        db_name = st.text_input("Database Name", value="issue_remediation")
        collection_name = st.text_input("Collection Name", value="issues")
        
        st.markdown("---")
    
    if not openai_key or not mongo_uri:
        st.warning("👈 Please provide OpenAI API Key and MongoDB connection string in the sidebar")
        return
    
    # Initialize clients
    openai_client = get_openai_client()
    mongo_client = get_mongo_client(mongo_uri)
    
    if not openai_client or not mongo_client:
        return
    
    db = mongo_client[db_name]
    collection = db[collection_name]
    
    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["🔍 Find Solution", "➕ Add to Knowledge Base", "📤 Upload CSV"])
    
    # Tab 1: Search for solution
    with tab1:
        st.header("Find Solution for Your Issue")
        
        user_issue = st.text_area("Describe your issue:", height=100, 
                                   placeholder="e.g., Application crashes when uploading large files")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            num_results = st.slider("Similar issues to show", 1, 5, 3)
        
        if st.button("🔍 Find Solution", type="primary"):
            if user_issue:
                with st.spinner("Searching knowledge base..."):
                    query_embedding = get_embedding(user_issue, openai_client)
                    similar_issues = vector_search(query_embedding, collection, limit=num_results)
                    
                    if similar_issues:
                        st.success("✅ Found similar issues!")
                        
                        # Generate solution
                        with st.spinner("Generating solution..."):
                            solution = generate_solution(user_issue, similar_issues, openai_client)
                        
                        st.markdown("### 💡 Recommended Solution")
                        st.success(solution)
                        
                        st.markdown("---")
                        st.markdown("### 📋 Similar Past Issues")
                        
                        for idx, issue in enumerate(similar_issues, 1):
                            with st.expander(f"#{idx} - {issue['category']} - Similarity: {issue['score']:.1%}", expanded=(idx==1)):
                                st.info(f"**🔴 Issue:**\n\n{issue['issue']}")
                                st.success(f"**✅ Solution:**\n\n{issue['solution']}")
                                st.caption(f"📅 Recorded: {issue['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    else:
                        st.warning("No similar issues found in the knowledge base.")
            else:
                st.error("Please describe your issue.")
    
    # Tab 2: Add new issue
    with tab2:
        st.header("Add New Issue to Knowledge Base")
        
        with st.form("add_issue_form"):
            new_issue = st.text_area("Issue Description:", height=100)
            new_solution = st.text_area("Solution:", height=150)
            new_category = st.selectbox("Category", 
                                        ["Network", "Database", "Application", "Security", "Performance", "Other"])
            
            if st.form_submit_button("➕ Add to Knowledge Base", type="primary"):
                if new_issue and new_solution:
                    with st.spinner("Adding to knowledge base..."):
                        add_issue_to_kb(new_issue, new_solution, new_category, collection, openai_client)
                        st.success("✅ Issue added successfully!")
                else:
                    st.error("Please fill in both issue and solution.")
    
    # Tab 3: Upload CSV
    with tab3:
        st.header("Bulk Upload from CSV")
        
        st.markdown("### CSV Format Required")
        st.markdown("Your CSV should have these columns: `issue`, `solution`, `category`")
        
        # Sample CSV download
        sample_data = {
            'issue': [
                'Application crashes when uploading files over 10MB',
                'Database connection timeout after 30 seconds',
                'Users unable to login with SSO'
            ],
            'solution': [
                'Increase max file upload size in config to 50MB and add chunked upload for large files',
                'Increase connection pool size and set timeout to 60 seconds in database config',
                'Check SSO certificate expiry and renew if needed, verify redirect URLs are whitelisted'
            ],
            'category': ['Application', 'Database', 'Security']
        }
        sample_df = pd.DataFrame(sample_data)
        
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_df.to_csv(index=False),
            file_name="sample_issues.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file:
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                
                # Try reading with different encodings
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
                
                if df.empty or len(df.columns) == 0:
                    st.error("❌ CSV file is empty or has no columns. Please check the file format.")
                    st.info("First line should be: issue,solution,category")
                else:
                    st.write("Preview:")
                    st.dataframe(df.head())
                    st.success(f"✅ Found {len(df)} rows and {len(df.columns)} columns")
                    
                    # Check required columns
                    required = ['issue', 'solution', 'category']
                    missing = [col for col in required if col not in df.columns]
                    if missing:
                        st.error(f"❌ Missing required columns: {', '.join(missing)}")
                        st.info(f"Your columns: {', '.join(df.columns.tolist())}")
                    else:
                        if st.button("📤 Upload to MongoDB", type="primary"):
                            uploaded_file.seek(0)
                            with st.spinner("Processing and uploading..."):
                                if load_csv_to_mongodb(uploaded_file, collection, openai_client):
                                    st.success(f"✅ Successfully uploaded {len(df)} records!")
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
                st.info("💡 Tips:\n- Make sure first line has: issue,solution,category\n- Save as UTF-8 encoding\n- No empty lines at the start\n- Use comma as separator")
    
    # Footer stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Stats")
    try:
        total_docs = collection.count_documents({})
        st.sidebar.metric("Total Issues in Knowledge Base", total_docs)
    except:
        pass

if __name__ == "__main__":
    main()